"""ADAPT-VQE algorithm framework, to solve electronic structure calculations.
It consists of constructing the ansatz as VQE iterations are performed.
From a set of operators, the most important one (stepiest energy gradient) versus
the current circuit is chosen. This operator is added to the ansatz, converted into
a set of gates and appended to the circuit. An VQE minimization is performed to
get a set of parameters that minimize the energy. The process is repeated n times
to end hopefully with a good ansatz for the studied molecule (or Hamiltonian).

Ref:
Grimsley, H.R., Economou, S.E., Barnes, E. et al.
An adaptive variational algorithm for exact molecular simulations on a quantum computer.
Nat Commun 10, 3007 (2019). https://doi.org/10.1038/s41467-019-10988-2
"""

from enum import Enum
from openfermion import commutator
from scipy.optimize import minimize

from qsdk.electronic_structure_solvers.vqe_solver import VQESolver
from qsdk.toolboxes.molecular_computation.frozen_orbitals import get_frozen_core
from qsdk.toolboxes.molecular_computation.molecular_data import MolecularData
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.ansatz_generator.adapt_uccgsd import ADAPTUCCGSD


class ADAPTAnsatze(Enum):
    """Enumeration of the adaptive ansatz circuits supported by ADAPT-VQE. """
    UCCGSD = 0


class ADAPTSolver:
    """ADAPT VQE class. This is basically a wrapper on top of VQE. Each iteration,
    the ansatz grows.

    Attributes:
        molecule (pyscf.gto.Mole): The molecular system.
        mean-field (optional): mean-field of molecular system.
        frozen_orbitals (list[int]): a list of indices for frozen orbitals.
            Default is the string "frozen_core", corresponding to the output
            of the function molecular_computation.frozen_orbitals.get_frozen_core.
        qubit_mapping (str): one of the supported qubit mapping identifiers
        ansatz (Ansatze): one of the supported ansatze.
        vqe_options:
        verbose (bool): Flag for verbosity of VQE.
     """

    def __init__(self, opt_dict):

        default_backend_options = {"target": "qulacs", "n_shots": None, "noise_model": None}
        default_options = {"molecule": None, "mean_field": None, "verbose": False,
                           "tol": 1e-3, "max_cycles": 15,
                           "qubit_mapping": "jw",
                           "ansatz": ADAPTAnsatze.UCCGSD,
                           "frozen_orbitals": "frozen_core",
                           "qubit_hamiltonian": None,
                           "n_spinorbitals": None,
                           "n_electrons": None,
                           "optimizer": self.LBFGSB_optimizer,
                           "backend_options": default_backend_options,
                           "up_then_down": False,
                           "verbose": True}

        # Initialize with default values
        self.__dict__ = default_options
        # Overwrite default values with user-provided ones, if they correspond to a valid keyword
        for k, v in opt_dict.items():
            if k in default_options:
                setattr(self, k, v)
            else:
                # TODO Raise a warning instead, that variable will not be used unless user made mods to code
                raise KeyError(f"Keyword :: {k}, not available in {self.__class__.__name__}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not (bool(self.molecule) ^ bool(self.qubit_hamiltonian)):
            raise ValueError(f"A molecule OR qubit Hamiltonian object must be provided when instantiating {self.__class__.__name__}.")

        self.converged = False
        self.iteration = 0

        self.optimal_energy = None
        self.optimal_var_params = None
        self.optimal_circuit = None
        self.builtin_ansatze = set(ADAPTAnsatze)

    @property
    def operators(self):
        return self.ansatz.operators

    @property
    def ferm_operators(self):
        return self.ansatz.ferm_operators

    def build(self):
        """Builds the underlying objects required to run the ADAPT-VQE algorithm. """

        # Building molecule data with a pyscf molecule.
        if self.molecule:
            # Build adequate mean-field.
            if not self.mean_field:
                self.mean_field = prepare_mf_RHF(self.molecule)

            # Same default as in vanilla VQE.
            if self.frozen_orbitals == "frozen_core":
                self.frozen_orbitals = get_frozen_core(self.molecule)

            # Compute qubit hamiltonian for the input molecular system.
            self.qemist_molecule = MolecularData(self.molecule, self.mean_field, self.frozen_orbitals)
            self.n_spinorbitals = 2 * self.qemist_molecule.n_orbitals
            self.n_electrons = self.qemist_molecule.n_electrons

            self.fermionic_hamiltonian = self.qemist_molecule.get_molecular_hamiltonian()
            self.qubit_hamiltonian = fermion_to_qubit_mapping(fermion_operator=self.fermionic_hamiltonian,
                                                              mapping=self.qubit_mapping,
                                                              n_spinorbitals=self.n_spinorbitals,
                                                              n_electrons=self.n_electrons,
                                                              up_then_down=self.up_then_down)
        else:
            assert(self.n_spinorbitals), "Expecting number of spin-orbitals (n_spinnorbitals) with a qubit_hamiltonian."
            assert(self.n_electrons), "Expecting number of electrons (n_electrons) with a qubit_hamiltonian."

        # Build / set ansatz circuit. Use user-provided circuit or built-in ansatz depending on user input.
        if type(self.ansatz) == ADAPTAnsatze:
            if self.ansatz == ADAPTAnsatze.UCCGSD:
                self.ansatz = ADAPTUCCGSD(n_spinorbitals=self.n_spinorbitals, n_electrons=self.n_electrons, mapping=self.qubit_mapping, up_then_down=self.up_then_down)
        else:
            raise ValueError(f"Unsupported ansatz. Expecting built-in adaptive ansatze:\n\t{self.builtin_ansatze}")

        # Build underlying VQE solver. Options remain consistent throughout the ADAPT cycles
        self.vqe_options = dict()
        self.vqe_options["qubit_hamiltonian"] = self.qubit_hamiltonian
        self.vqe_options["ansatz"] = self.ansatz
        self.vqe_options["optimizer"] = self.optimizer

        self.vqe_solver = VQESolver(self.vqe_options)
        self.vqe_solver.build()

        self.pool_operators, self.fermionic_operators = self.ansatz.get_pool()

        # Getting commutators to compute gradients:
        # \frac{\partial E}{\partial \theta_n} = \langle \psi | [\hat{H}, A_n] | \psi \rangle
        self.pool_commutators = [commutator(self.qubit_hamiltonian, element) for element in self.pool_operators]

    def simulate(self):
        """Performs the ADAPT cycles. Each iteration, a VQE minimization is done. """

        energies = list()
        params = self.vqe_solver.ansatz.var_params

        # Construction of the ansatz. self.max_cycles terms are added, unless
        # all operator gradients are less than self.tol.
        while self.iteration < self.max_cycles:
            self.iteration += 1
            print(f"Iteration {self.iteration} of ADAPT-VQE.")

            pool_select = self.rank_pool(self.pool_commutators, self.vqe_solver.ansatz.circuit,
                                    backend=self.vqe_solver.backend, tolerance=self.tol)

            # If pool selection returns an operator that changes the energy by
            # more than self.tol. Else, the loop is complete and the energy is
            # considered as converged.
            if pool_select > -1:

                # Adding a new operator + initializing its parameters to 0.
                # Previous parameters are kept as they were.
                params += [0.]
                self.vqe_solver.ansatz.add_operator(self.pool_operators[pool_select], self.fermionic_operators[pool_select])
                self.vqe_solver.initial_var_params = params

                # Performs a VQE simulation and append the energy to a list.
                # Also, forcing params to be a list to make it easier to append
                # new parameters. The behavior with a np.array is multiplication
                # with broadcasting (not wanted).
                opt_energy, params = self.vqe_solver.simulate()
                params = list(params)
                energies.append(opt_energy)
            else:
                self.converged = True
                break

        return energies

    def rank_pool(self, pool_commutators, circuit, backend, tolerance=1e-3):
        """Rank pool of operators with a specific circuit.

        Args:
            pool_commutators (QubitOperator): Commutator [H, operator] for each generator.
            circuit (agnostic_simulator.Circuit): Circuit for measuring each commutator.
            backend (angostic_simulator.Simulator): Backend to measure expectation values.
            tolerance (float): Minimum value for gradient to be considered.

        Returns:
            int: Index of the operators with the highest gradient. If it is not
                bigger than tolerance, returns -1.
        """

        gradient = [abs(backend.get_expectation_value(element, circuit)) for element in pool_commutators]
        max_partial = max(gradient)

        if self.verbose:
            print(f'LARGEST PARTIAL DERIVATIVE: {max_partial :4E} \t[{gradient.index(max_partial)}]')

        return gradient.index(max_partial) if max_partial >= tolerance else -1

    def get_resources(self):
        """Returns resources currently used in underlying VQE. """

        return self.vqe_solver.get_resources()

    def LBFGSB_optimizer(self, func, var_params):
        """Default optimizer for ADAPT-VQE. According to our in-house experiments,
        this one is more robust for ADAPT layer.
        """

        result = minimize(func, var_params, method="L-BFGS-B",
            options={"disp": False, "maxiter": 100, 'gtol': 1e-10, 'iprint': -1})

        self.optimal_var_params = result.x
        self.optimal_energy = result.fun

        # Only defined at the of ADAPT cycles. If not, the circuit is rebuilt
        # everytime and each ADAPT cycle becomes slow.
        if self.converged or self.iteration == self.max_cycles:
            self.ansatz.build_circuit(self.optimal_var_params)
            self.optimal_circuit = self.vqe_solver.ansatz.circuit

        if self.verbose:
            print(f"\t\tOptimal ADAPT-VQE energy: {self.optimal_energy}")
            print(f"\t\tOptimal VQE variational parameters: {self.optimal_var_params}")
            print(f"\t\tNumber of Function Evaluations : {result.nfev}")

        return result.fun, result.x