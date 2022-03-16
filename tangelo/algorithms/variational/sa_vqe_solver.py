# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the variational quantum eigensolver (VQE) algorithm to solve
electronic structure calculations.
"""

import warnings
import itertools

from enum import Enum
import numpy as np

from tangelo.helpers.utils import HiddenPrints
from tangelo.linq import Simulator, Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates
from tangelo.toolboxes.operators import count_qubits, FermionOperator, qubitop_to_qubitham
from tangelo.toolboxes.qubit_mappings import statevector_mapping
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.ansatz_generator.ansatz import Ansatz
from tangelo.toolboxes.ansatz_generator import UCCSD, RUCC, HEA, UpCCGSD, QMF, QCC, VSQS, VariationalCircuitAnsatz
from tangelo.toolboxes.ansatz_generator.uccgd import UCCGD
from tangelo.toolboxes.ansatz_generator.penalty_terms import combined_penalty
from tangelo.toolboxes.post_processing.bootstrapping import get_resampled_frequencies
from tangelo.algorithms.variational import BuiltInAnsatze


class SA_VQESolver:
    r"""Solve the electronic structure problem for a molecular system by using
    the variational quantum eigensolver (VQE) algorithm.

    This algorithm evaluates the energy of a molecular system by performing
    classical optimization over a parametrized ansatz circuit.

    Users must first set the desired options of the VQESolver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, ansatz...). They are then able to call any of
    the energy_estimation, simulate, or get_rdm methods. In particular, simulate
    runs the VQE algorithm, returning the optimal energy found by the classical
    optimizer.

    Attributes:
        molecule (SecondQuantizedMolecule) : the molecular system.
        qubit_mapping (str) : one of the supported qubit mapping identifiers.
        ansatz (Ansatze) : one of the supported ansatze.
        optimizer (function handle): a function defining the classical optimizer
            and its behavior.
        initial_var_params (str or array-like) : initial value for the classical
            optimizer.
        backend_options (dict) : parameters to build the tangelo.linq Simulator
            class.
        penalty_terms (dict): parameters for penalty terms to append to target
            qubit Hamiltonian (see penaly_terms for more details).
        ansatz_options (dict): parameters for the given ansatz (see given ansatz
            file for details).
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False has alternating
                spin up/down ordering.
        qubit_hamiltonian (QubitOperator-like): Self-explanatory.
        verbose (bool): Flag for VQE verbosity.
        spins (list): list of spins to optimize states over
        ref_states (list): The vector occupations of the reference configurations
        weights (array): The weights of the occupations
    """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        default_options = {"molecule": None,
                           "qubit_mapping": "jw", "ansatz": BuiltInAnsatze.UCCGD,
                           "optimizer": self._default_optimizer,
                           "initial_var_params": None,
                           "backend_options": default_backend_options,
                           "penalty_terms": None,
                           "ansatz_options": dict(),
                           "up_then_down": False,
                           "qubit_hamiltonian": None,
                           "verbose": False,
                           "ref_states": None,
                           "weights": None}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                raise KeyError(f"Keyword :: {k}, not available in VQESolver")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

        # The QCC ansatz requires up_then_down=True when mapping="jw"
        if isinstance(self.ansatz, BuiltInAnsatze):
            if self.ansatz == BuiltInAnsatze.QCC and self.qubit_mapping.lower() == "jw" and not self.up_then_down:
                warnings.warn("The QCC ansatz requires spin-orbital ordering to be all spin-up "
                              "first followed by all spin-down for the JW mapping.", RuntimeWarning)
                self.up_then_down = True

        self.optimal_energy = None
        self.optimal_var_params = None
        self.builtin_ansatze = set(BuiltInAnsatze)

    def build(self):
        """Build the underlying objects required to run the VQE algorithm
        afterwards.
        """

        if isinstance(self.ansatz, Circuit):
            self.ansatz = VariationalCircuitAnsatz(self.ansatz)

        # Building VQE with a molecule as input.
        if self.molecule:

            # Compute qubit hamiltonian for the input molecular system
            qubit_op = fermion_to_qubit_mapping(fermion_operator=self.molecule.fermionic_hamiltonian,
                                                mapping=self.qubit_mapping,
                                                n_spinorbitals=self.molecule.n_active_sos,
                                                n_electrons=self.molecule.n_active_electrons,
                                                up_then_down=self.up_then_down,
                                                spin=self.molecule.spin)

            self.core_constant, self.oneint, self.twoint = self.molecule.get_active_space_integrals()

            self.qubit_hamiltonian = qubitop_to_qubitham(qubit_op, self.qubit_mapping, self.up_then_down)

            if self.penalty_terms:
                pen_ferm = combined_penalty(self.molecule.n_active_mos, self.penalty_terms)
                pen_qubit = fermion_to_qubit_mapping(fermion_operator=pen_ferm,
                                                     mapping=self.qubit_mapping,
                                                     n_spinorbitals=self.molecule.n_active_sos,
                                                     n_electrons=self.molecule.n_active_electrons,
                                                     up_then_down=self.up_then_down,
                                                     spin=self.molecule.spin)
                pen_qubit = qubitop_to_qubitham(pen_qubit, self.qubit_hamiltonian.mapping, self.qubit_hamiltonian.up_then_down)
                self.qubit_hamiltonian += pen_qubit

            # Verification of system compatibility with UCC1 or UCC3 circuits.
            if self.ansatz in [BuiltInAnsatze.UCC1, BuiltInAnsatze.UCC3]:
                # Mapping should be JW because those ansatz are chemically inspired.
                if self.qubit_mapping.upper() != "JW":
                    raise ValueError("Qubit mapping must be JW for {} ansatz.".format(self.ansatz))
                # They are encoded with this convention.
                if not self.up_then_down:
                    raise ValueError("Parameter up_then_down must be set to True for {} ansatz.".format(self.ansatz))
                # Only HOMO-LUMO systems are relevant.
                if count_qubits(self.qubit_hamiltonian) != 4:
                    raise ValueError("The system must be reduced to a HOMO-LUMO problem for {} ansatz.".format(self.ansatz))

            # Build / set ansatz circuit. Use user-provided circuit or built-in ansatz depending on user input.
            if isinstance(self.ansatz, BuiltInAnsatze):
                if self.ansatz == BuiltInAnsatze.UCCSD:
                    self.ansatz = UCCSD(self.molecule, self.qubit_mapping, self.up_then_down, self.molecule.spin)
                elif self.ansatz == BuiltInAnsatze.UCC1:
                    self.ansatz = RUCC(1)
                elif self.ansatz == BuiltInAnsatze.UCC3:
                    self.ansatz = RUCC(3)
                elif self.ansatz == BuiltInAnsatze.HEA:
                    self.ansatz = HEA(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.UpCCGSD:
                    self.ansatz = UpCCGSD(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.QMF:
                    self.ansatz = QMF(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.QCC:
                    self.ansatz = QCC(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.VSQS:
                    self.ansatz = VSQS(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
                elif self.ansatz == BuiltInAnsatze.UCCGD:
                    self.ansatz = UCCGD(self.molecule, self.qubit_mapping, up_then_down=self.up_then_down)
                else:
                    raise ValueError(f"Unsupported ansatz. Built-in ansatze:\n\t{self.builtin_ansatze}")
            elif not isinstance(self.ansatz, Ansatz):
                print(type(self.ansatz))
                raise TypeError(f"Invalid ansatz dataype. Expecting instance of Ansatz class, or one of built-in options:\n\t{self.builtin_ansatze}")

            self.ansatz.default_reference_state = "None"
            self.reference_circuits = list()
            for ref_state in self.ref_states:
                vec_to_map = np.concatenate((ref_state[::2], ref_state[1::2])) if self.up_then_down else ref_state
                if self.qubit_mapping.lower() == "scbk":
                    mapped_state = statevector_mapping.do_scbk_transform(vec_to_map, self.molecule.n_active_sos)
                elif self.qubit_mapping.lower() == "bk":
                    mapped_state = statevector_mapping.do_bk_transform(vec_to_map)
                else:
                    mapped_state = vec_to_map
                self.reference_circuits.append(statevector_mapping.vector_to_circuit(mapped_state, self.qubit_mapping))
            self.n_states = len(self.ref_states)
            if self.weights is None:
                self.weights = np.ones(self.n_states)/self.n_states
            else:
                if len(self.weights) != self.n_states:
                    raise ValueError("Number of elements in weights must equal the number of ref_configs")
                self.weights = np.array(self.weights)
                self.weights = self.weights/sum(self.weights)

        # Building with a qubit Hamiltonian.
        elif self.ansatz in [BuiltInAnsatze.HEA, BuiltInAnsatze.VSQS]:
            if self.ansatz == BuiltInAnsatze.HEA:
                self.ansatz = HEA(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
            elif self.ansatz == BuiltInAnsatze.VSQS:
                self.ansatz = VSQS(self.molecule, self.qubit_mapping, self.up_then_down, **self.ansatz_options)
        elif not isinstance(self.ansatz, Ansatz):
            raise TypeError(f"Invalid ansatz dataype. Expecting a custom Ansatz (Ansatz class).")

        # Set ansatz initial parameters (default or use input), build corresponding ansatz circuit
        self.initial_var_params = self.ansatz.set_var_params(self.initial_var_params)
        self.ansatz.build_circuit()

        # Quantum circuit simulation backend options
        self.backend = Simulator(target=self.backend_options["target"], n_shots=self.backend_options["n_shots"],
                                 noise_model=self.backend_options["noise_model"])

    def simulate(self):
        """Run the VQE algorithm, using the ansatz, classical optimizer, initial
        parameters and hardware backend built in the build method.
        """
        if not (self.ansatz and self.backend):
            raise RuntimeError("No ansatz circuit or hardware backend built. Have you called VQESolver.build ?")
        optimal_energy, optimal_var_params = self.optimizer(self.energy_estimation, self.initial_var_params)

        self.optimal_var_params = optimal_var_params
        self.optimal_energy = optimal_energy
        self.ansatz.build_circuit(self.optimal_var_params)
        self.optimal_circuit = self.ansatz.circuit
        self.rdms = list()
        for reference_circuit in self.reference_circuits:
            self.rdms.append(self.get_rdm(self.optimal_var_params, ref_state=reference_circuit))
        return self.optimal_energy

    def get_resources(self):
        """Estimate the resources required by VQE, with the current ansatz. This
        assumes "build" has been run, as it requires the ansatz circuit and the
        qubit Hamiltonian. Return information that pertains to the user, for the
        purpose of running an experiment on a classical simulator or a quantum
        device.
        """

        resources = dict()
        resources["qubit_hamiltonian_terms"] = len(self.qubit_hamiltonian.terms)
        resources["circuit_width"] = self.ansatz.circuit.width
        resources["circuit_gates"] = self.ansatz.circuit.size
        # For now, only CNOTs supported.
        resources["circuit_2qubit_gates"] = self.ansatz.circuit.counts.get("CNOT", 0)
        resources["circuit_var_gates"] = len(self.ansatz.circuit._variational_gates)
        resources["vqe_variational_parameters"] = len(self.initial_var_params)
        return resources

    def energy_estimation(self, var_params):
        """Estimate energy using the given ansatz, qubit hamiltonian and compute
        backend. Keeps track of optimal energy and variational parameters along
        the way.

        Args:
             var_params (numpy.array or str): variational parameters to use for
                VQE energy evaluation.

        Returns:
             float: energy computed by VQE using the ansatz and input
                variational parameters.
        """

        # Update variational parameters, compute energy using the hardware backend
        self.ansatz.update_var_params(var_params)
        energy = 0
        self.state_energies = list()
        for i, reference_circuit in enumerate(self.reference_circuits):
            state_energy = self.backend.get_expectation_value(self.qubit_hamiltonian, reference_circuit + self.ansatz.circuit)
            energy += state_energy*self.weights[i]
            self.state_energies.append(state_energy)

        if self.verbose:
            print(f"\tEnergy = {energy:.7f} ")

        return energy

    def get_rdm(self, var_params,  ref_state=Circuit(), resample=False, sum_spin=True):
        """Compute the 1- and 2- RDM matrices using the VQE energy evaluation.
        This method allows to combine the DMET problem decomposition technique
        with the VQE as an electronic structure solver. The RDMs are computed by
        using each fermionic Hamiltonian term, transforming them and computing
        the elements one-by-one. Note that the Hamiltonian coefficients will not
        be multiplied as in the energy evaluation. The first element of the
        Hamiltonian is the nuclear repulsion energy term, not the Hamiltonian
        term.

        Args:
            var_params (numpy.array or list): variational parameters to use for
                rdm calculation
            ref_state (Circuit): The circuit to use for the reference_state
            resample (bool): Whether to resample saved frequencies. get_rdm with
                savefrequencies=True must be called or a dictionary for each
                qubit terms' frequencies must be set to self.rdm_freq_dict
            sum_spin (bool): If True, the spin-summed 1-RDM and 2-RDM will be
                returned. If False, the full 1-RDM and 2-RDM will be returned.

        Returns:
            (numpy.array, numpy.array): One & two-particle spin summed RDMs if
                sumspin=True or the full One & two-Particle RDMs if
                sumspin=False.
        """

        self.ansatz.update_var_params(var_params)

        # Initialize the RDM arrays
        n_mol_orbitals = self.molecule.n_active_mos
        n_spin_orbitals = self.molecule.n_active_sos
        rdm1_spin = np.zeros((n_spin_orbitals,) * 2, dtype=complex)
        rdm2_spin = np.zeros((n_spin_orbitals,) * 4, dtype=complex)

        # If resampling is requested, check that a previous savefrequencies run has been called
        if resample:
            if hasattr(self, "rdm_freq_dict"):
                qb_freq_dict = self.rdm_freq_dict
                resampled_expect_dict = dict()
            else:
                raise AttributeError("Need to run RDM calculation with savefrequencies=True")
        else:
            qb_freq_dict = dict()
            qb_expect_dict = dict()

        # Loop over each element of Hamiltonian (non-zero value)
        for key in self.molecule.fermionic_hamiltonian.terms:
            # Ignore constant / empty term
            if not key:
                continue

            # Assign indices depending on one- or two-body term
            length = len(key)
            if (length == 2):
                iele, jele = (int(ele[0]) for ele in tuple(key[0:2]))
            elif (length == 4):
                iele, jele, kele, lele = (int(ele[0]) for ele in tuple(key[0:4]))

            # Create the Hamiltonian with the correct key (Set coefficient to one)
            hamiltonian_temp = FermionOperator(key)

            # Obtain qubit Hamiltonian
            qubit_hamiltonian2 = fermion_to_qubit_mapping(fermion_operator=hamiltonian_temp,
                                                          mapping=self.qubit_mapping,
                                                          n_spinorbitals=self.molecule.n_active_sos,
                                                          n_electrons=self.molecule.n_active_electrons,
                                                          up_then_down=self.up_then_down,
                                                          spin=self.molecule.spin)
            qubit_hamiltonian2.compress()

            # Run through each qubit term separately, use previously calculated result for the qubit term or
            # calculate and save results for that qubit term
            opt_energy2 = 0.
            for qb_term, qb_coef in qubit_hamiltonian2.terms.items():
                if qb_term:
                    if qb_term not in qb_freq_dict:
                        if resample:
                            warnings.warn(f"Warning: rerunning circuit for missing qubit term {qb_term}")
                        basis_circuit = Circuit(measurement_basis_gates(qb_term))
                        full_circuit = ref_state + self.ansatz.circuit + basis_circuit
                        qb_freq_dict[qb_term], _ = self.backend.simulate(full_circuit)
                    if resample:
                        if qb_term not in resampled_expect_dict:
                            resampled_freq_dict = get_resampled_frequencies(qb_freq_dict[qb_term], self.backend.n_shots)
                            resampled_expect_dict[qb_term] = self.backend.get_expectation_value_from_frequencies_oneterm(qb_term, resampled_freq_dict)
                        expectation = resampled_expect_dict[qb_term]
                    else:
                        if qb_term not in qb_expect_dict:
                            qb_expect_dict[qb_term] = self.backend.get_expectation_value_from_frequencies_oneterm(qb_term, qb_freq_dict[qb_term])
                        expectation = qb_expect_dict[qb_term]
                    opt_energy2 += qb_coef * expectation
                else:
                    opt_energy2 += qb_coef

            # Put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                rdm1_spin[iele, jele] += opt_energy2
            elif length == 4:
                rdm2_spin[iele, lele, jele, kele] += opt_energy2

        # save rdm frequency dictionary
        self.rdm_freq_dict = qb_freq_dict

        if sum_spin:
            rdm1_np = np.zeros((n_mol_orbitals,) * 2, dtype=np.complex128)
            rdm2_np = np.zeros((n_mol_orbitals,) * 4, dtype=np.complex128)

            # Construct spin-summed 1-RDM
            for i, j in itertools.product(range(n_spin_orbitals), repeat=2):
                rdm1_np[i//2, j//2] += rdm1_spin[i, j]

            # Construct spin-summed 2-RDM
            for i, j, k, l in itertools.product(range(n_spin_orbitals), repeat=4):
                rdm2_np[i//2, j//2, k//2, l//2] += rdm2_spin[i, j, k, l]

            return rdm1_np, rdm2_np

        return rdm1_spin, rdm2_spin

    def _compute_energy(self, onerdm, twordm):
        """Calculate the fragment energy.

        Args:
            onerdm (numpy.array): one-particle reduced density matrix (float).
            twordm (numpy.array): two-particle reduced density matrix (float).
            fock_frag_copy (numpy.array): Fock matrix with the chemical
                potential subtracted (float).

        Returns:
            float: Fragment energy (fragment_energy).
            float: Total energy for fragment using RDMs (total_energy_rdm).
            numpy.array: One-particle RDM for a fragment (one_rdm, float).
        """

        # Calculate the total energy based on RDMs
        total_energy_rdm = np.einsum("ij,ij->", self.oneint, onerdm) + 0.5 * np.einsum("ijkl,iljk->", self.twoint, twordm)

        return total_energy_rdm

    def _default_optimizer(self, func, var_params):
        """Function used as a default optimizer for VQE when user does not
        provide one. Can be used as an example for users who wish to provide
        their custom optimizer.

        Should set the attributes "optimal_var_params" and "optimal_energy" to
        ensure the outcome of VQE is captured at the end of classical
        optimization, and can be accessed in a standard way.

        Args:
            func (function handle): The function that performs energy
                estimation. This function takes var_params as input and returns
                a float.
            var_params (list): The variational parameters (float64).

        Returns:
            float: The optimal energy found by the optimizer.
            list of floats: Optimal parameters
        """

        from scipy.optimize import minimize

        with HiddenPrints():
            result = minimize(func, var_params, method="SLSQP",
                              options={"disp": True, "maxiter": 2000, "eps": 1e-7, "ftol": 1e-7})

        if self.verbose:
            print(f"VQESolver optimization results:")
            print(f"\tOptimal VQE energy: {result.fun}")
            print(f"\tOptimal VQE variational parameters: {result.x}")
            print(f"\tNumber of Iterations : {result.nit}")
            print(f"\tNumber of Function Evaluations : {result.nfev}")
            print(f"\tNumber of Gradient Evaluations : {result.njev}")

        return result.fun, result.x
