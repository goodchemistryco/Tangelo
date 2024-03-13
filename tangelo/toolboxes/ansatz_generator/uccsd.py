# Copyright 2023 Good Chemistry Company.
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

"""This module defines the UCCSD ansatz class. It provides a chemically inspired
ansatz and is an implementation of the classical unitary CCSD ansatz. Single and
double excitation determinants, in accordance with the system number of electron
and spin, are considered. For more information about this ansatz, see references
below.

Refs:
    - P.Kl. Barkoutsos, J.F. Gonthier, I. Sokolov, N. Moll, G. Salis, A. Fuhrer,
        M. Ganzhorn, D.J. Egger, M. Troyer, A. Mezzacapo, S. Filipp, and
        I. Tavernelli. Phys. Rev. A 98, 022322 (2018).
    - I.O. Sokolov, P.Kl. Barkoutsos, P.J. Ollitrault, D. Greenberg, J. Rice,
        M. Pistoia, and I. Tavernelli. J. Chem. Phys. 152, 124107 (2020).
    - Y. Shen, X. Zhang, S. Zhang, J.N. Zhang, M.H. Yung, and K. Kim.
        Physical Review A 95, 020501 (2017).
"""

from typing import Union
from copy import deepcopy

import numpy as np

from tangelo import SecondQuantizedMolecule
from tangelo.linq import Circuit
from tangelo.helpers.utils import is_package_installed
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from tangelo.toolboxes.molecular_computation import IntegralSolverPySCF
from tangelo.toolboxes.operators import QubitOperator
from .ansatz import Ansatz
from .ansatz_utils import exp_pauliword_to_gates
from ._unitary_cc_openshell import uccsd_openshell_paramsize, uccsd_openshell_generator
from ._unitary_cc import uccsd_singlet_generator


class UCCSD(Ansatz):
    """This class implements the UCCSD ansatz. Currently, closed-shell and
    restricted open-shell UCCSD are supported. This implies that the mean-field
    is computed with the RHF or ROHF reference integrals.

    Args:
        molecule (SecondQuantizedMolecule) : The molecular system.
        mapping (str) : one of the supported qubit mapping identifiers. Default,
            "jw".
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False (i.e. has
            alternating spin up/down ordering).
        reference_state (string): The reference state id for the ansatz. The
            supported reference states are stored in the supported_reference_state
            attributes. Default, "HF".
    """

    def __init__(self, molecule: SecondQuantizedMolecule, mapping: str = "JW", up_then_down: bool = False, spin: Union[int, None] = None, reference_state="HF"):

        self.molecule: SecondQuantizedMolecule = molecule
        self.n_spinorbitals = molecule.n_active_sos
        self.n_electrons = molecule.n_active_electrons
        self.spin = molecule.active_spin if spin is None else spin
        self.mapping = mapping
        self.up_then_down = up_then_down

        # Later: refactor to handle various flavors of UCCSD
        if not self.molecule.uhf and self.n_spinorbitals % 2 != 0:
            raise ValueError("The total number of spin-orbitals should be even when reference is RHF or ROHF.")

        # choose open-shell uccsd if spin not zero, else choose singlet ccsd
        if self.spin != 0 or self.molecule.uhf:
            self.n_alpha, self.n_beta = self.molecule.n_active_ab_electrons
            if self.molecule.uhf:
                self.n_orb_a = self.molecule.n_active_mos[0]
                self.n_orb_b = self.molecule.n_active_mos[1]
            else:
                self.n_orb_a = self.n_spinorbitals//2
                self.n_orb_b = self.n_spinorbitals//2
            self.n_singles, self.n_doubles, _, _, _, _, _ = uccsd_openshell_paramsize(self.n_alpha, self.n_beta, self.n_orb_a, self.n_orb_b)
            # set total number of parameters
            # TODO UHF or ROHF symmetry adapated.
            self.n_var_params = self.n_singles + self.n_doubles
            self.params2keep = list(range(self.n_var_params))
        else:
            self.n_spatial_orbitals = self.n_spinorbitals // 2
            self.n_occupied = int(np.ceil(self.n_electrons / 2))
            self.n_virtual = self.n_spatial_orbitals - self.n_occupied
            self.n_singles = self.n_occupied * self.n_virtual
            self.n_doubles = self.n_singles * (self.n_singles + 1) // 2
            # Determine parameters to keep depending on irreducible representation. If no symmetry, all are kept
            self._singlet_symmetry_mask()
            # Set total number of possibly symmetry reduced parameters
            self.n_var_params = len(self.params2keep)

        # Set number of possible var params, can be higher than n_var_params
        self.n_full_var_params = self.n_singles + self.n_doubles

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF", "zero"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random", "mp2"}

        # Default initial parameters for initialization
        self.var_params_default = "mp2"
        self.reference_state = reference_state

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """Set values for variational parameters, such as zeros, random numbers,
        MP2 (...), providing some keywords for users, and also supporting direct
        user input (list or numpy array). Return the parameters so that
        workflows such as VQE can retrieve these values.
        """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            var_params = var_params.lower()
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = 2.e-1 * (np.random.random((self.n_var_params,)) - 0.5)
            elif var_params == "mp2":
                initial_var_params = self._compute_mp2_params()[self.params2keep]
        else:
            initial_var_params = np.array(var_params)
            if initial_var_params.size != self.n_var_params:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but "
                                 f"received {initial_var_params.size}.")

        self.var_params = np.zeros(self.n_full_var_params)
        self.var_params[self.params2keep] = initial_var_params
        return list(initial_var_params)

    def prepare_reference_state(self):
        """Returns circuit preparing the reference state of the ansatz (e.g
        prepare reference wavefunction with HF, multi-reference state, etc).
        These preparations must be consistent with the transform used to obtain
        the qubit operator.
        """

        if self.reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_spinorbitals,
                                         n_electrons=self.n_electrons,
                                         mapping=self.mapping,
                                         up_then_down=self.up_then_down,
                                         spin=self.spin)
        elif self.reference_state == "zero":
            return Circuit()

    def build_circuit(self, var_params=None):
        """Build and return the quantum circuit implementing the state
        preparation ansatz (with currently specified initial_state and
        var_params).
        """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit_operator() if (self.spin == 0 and not self.molecule.uhf) else self._get_openshell_qubit_operator()

        # Prepend reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Obtain quantum circuit through trivial trotterization of the qubit operator
        # Keep track of the order in which pauli words have been visited for fast subsequent parameter updates
        pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
        pauli_words_gates = []
        self.pauli_to_angles_mapping = dict()
        for i, (pauli_word, coef) in enumerate(pauli_words):
            pauli_words_gates += exp_pauliword_to_gates(pauli_word, coef)
            self.pauli_to_angles_mapping[pauli_word] = i

        uccsd_circuit = Circuit(pauli_words_gates)
        # skip over the reference state circuit if it is empty
        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + uccsd_circuit
        else:
            self.circuit = uccsd_circuit

    def update_var_params(self, var_params):
        """Shortcut: set value of variational parameters in the already-built
        ansatz circuit member. Preferable to rebuilt your circuit from scratch,
        which can be an involved process.
        """

        self.set_var_params(var_params)

        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit_operator() if (self.spin == 0 and not self.molecule.uhf) else self._get_openshell_qubit_operator()

        # If qubit operator terms have changed, rebuild circuit. Else, simply update variational gates directly
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            self.build_circuit(var_params)
        else:
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                self.circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef

    def _get_singlet_qubit_operator(self):
        """Construct UCCSD FermionOperator for current variational parameters,
        and translate to QubitOperator via relevant qubit mapping.

        Returns:
            QubitOperator: qubit-encoded elements of the UCCSD ansatz.
        """
        if not hasattr(self, "param_dict"):
            self._singlet_symmetry_mask()

        qubit_op = QubitOperator()
        for key in self.params2keep:
            # Obtain corresponding excitation
            value = self.param_dict[key]
            # Somehow the operator_dict[value] gets modified by multiplication
            qubit_op += fermion_to_qubit_mapping(fermion_operator=deepcopy(self.operator_dict[value])*self.var_params[key],
                                                 mapping=self.mapping,
                                                 n_spinorbitals=self.n_spinorbitals,
                                                 n_electrons=self.n_electrons,
                                                 up_then_down=self.up_then_down,
                                                 spin=self.spin)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op

    def _singlet_symmetry_mask(self):
        """Generate the mapping between the full packed amplitude list and those retained for the circuit"""
        self.param_dict, self.operator_dict = uccsd_singlet_generator(self.n_spinorbitals, self.n_electrons)

        # If symmetry is present, only keep parameters that have the original irreduciple representation.
        if self.molecule.symmetry:

            # Obtain the symmetry ids for each molecular orbital
            mo_ids = [self.molecule.mo_symm_ids[i] for i in self.molecule.active_occupied+self.molecule.active_virtual]

            # Go through each excitation operator and determine the result irrep.
            params2keep = []
            for key, value in self.param_dict.items():
                # Generate the reference configuration. The irrep is always 0 for a singlet
                ref_occ = [list(range(self.n_occupied)), list(range(self.n_occupied))]
                if len(value) == 2:
                    ref_occ[0].remove(value[1])
                    ref_occ[0] += [value[0]]
                    irrep = 0
                    for s in range(2):
                        for occ in ref_occ[s]:
                            irrep ^= mo_ids[occ]
                    # If the irrep stays 0, that parameter can contribute
                    if irrep == 0:
                        params2keep += [key]
                else:
                    ref_occ[0].remove(value[1])
                    ref_occ[0] += [value[0]]
                    ref_occ[1].remove(value[3])
                    ref_occ[1] += [value[2]]
                    irrep = 0
                    for s in range(2):
                        for occ in ref_occ[s]:
                            irrep ^= mo_ids[occ]
                    # If the irrep stays 0, that parameter can contribute
                    if irrep == 0:
                        params2keep += [key]
            self.params2keep = params2keep

        # Keep all parameters
        else:
            self.params2keep = list(range(len(self.param_dict)))

    def _get_openshell_qubit_operator(self):
        """Construct open-shell UCCSD FermionOperator for current variational
        parameters, and translate to QubitOperator via relevant qubit mapping.

        Returns:
            QubitOperator: qubit-encoded elements of the UCCSD ansatz.
        """
        fermion_op = uccsd_openshell_generator(self.var_params,
                                               self.n_alpha,
                                               self.n_beta,
                                               self.n_orb_a,
                                               self.n_orb_b)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.mapping,
                                            n_spinorbitals=self.n_spinorbitals,
                                            n_electrons=self.n_electrons,
                                            up_then_down=self.up_then_down,
                                            spin=self.spin)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op

    def _compute_mp2_params(self):
        """Compute the MP2 initial variational parameters. Only double
        excitation amplitudes are retrieved from an MP2 calculation (singles
        set to a small number close to 0).

        Returns:
            list of float: The initial variational parameters.
        """

        if not is_package_installed("pyscf"):
            supported_initial_var_params = self.supported_initial_var_params.copy()
            supported_initial_var_params.remove("mp2")
            raise ValueError(f"PySCF is required for MP2 initial parameters in {self.__class__.__name__}.\n"
                             f"Other supported keywords for initializing variational parameters are {supported_initial_var_params}. "
                             f"An array of floats of length {self.n_var_params} can also be provided.\n"
                             f"The above keywords or array can also be provided to variational algorithms through the 'initial_var_params' keyword.")

        # Import here to solve an AttributeError: partially initialized module
        # tangelo.toolboxes.ansatz_generator' has no attribute 'UCCSD'
        # (most likely due to a circular import).
        from tangelo.algorithms.classical.mp2_solver import MP2Solver

        if not isinstance(self.molecule.solver, IntegralSolverPySCF):
            pymol = SecondQuantizedMolecule(self.molecule.xyz, self.molecule.q, self.molecule.spin, basis=self.molecule.basis,
                                            ecp=self.molecule.ecp, symmetry=self.molecule.symmetry, uhf=self.molecule.uhf,
                                            frozen_orbitals=self.molecule.frozen_orbitals)
        else:
            pymol = self.molecule

        mp2 = MP2Solver(pymol)
        mp2.simulate()

        return np.array(mp2.get_mp2_amplitudes())
