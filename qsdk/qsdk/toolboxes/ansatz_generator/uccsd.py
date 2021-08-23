""" This module defines the UCCSD ansatz class. It provides a chemically inspired ansatzs
    and is an implementation of the classical unitary CCSD ansatz. Single and double excitation
    determinants, in accordance with the system number of electron and spin, are considered.
    For more information about this ansatz, see references below.

    Refs:
        * P.Kl. Barkoutsos, J.F. Gonthier, I. Sokolov, N. Moll, G. Salis, A. Fuhrer, M. Ganzhorn,
          D.J. Egger, M. Troyer, A. Mezzacapo, S. Filipp, and I. Tavernelli, Phys. Rev. A 98, 022322 (2018).
        * I.O. Sokolov, P.Kl. Barkoutsos, P.J. Ollitrault, D. Greenberg, J. Rice, M. Pistoia,
          and I. Tavernelli, J. Chem. Phys. 152, 124107 (2020).
        * Y. Shen, X. Zhang, S. Zhang, J.N. Zhang, M.H. Yung, and K. Kim, Physical Review A 95, 020501 (2017).
"""

import itertools
import numpy as np
from pyscf import mp

from agnostic_simulator import Circuit

from .ansatz import Ansatz
from .ansatz_utils import pauliword_to_circuit
from ._unitary_cc import uccsd_singlet_generator
from ._unitary_cc_openshell import uccsd_openshell_paramsize, uccsd_openshell_generator
from qsdk.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from qsdk.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF


class UCCSD(Ansatz):
    """ This class implements the UCCSD ansatz. Currently, only closed-shell UCCSD is supported.
     This implies that the mean-field is computed with the RHF reference integrals. """

    def __init__(self, n_spinorbitals, n_electrons, spin=0, mapping="JW", up_then_down=False):

        self.n_spinorbitals = n_spinorbitals
        self.n_electrons = n_electrons
        self.spin = spin
        self.mapping = mapping
        self.up_then_down = up_then_down

        # Later: refactor to handle various flavors of UCCSD
        if n_spinorbitals % 2 != 0:
            raise ValueError('The total number of spin-orbitals should be even.')

        # choose open-shell uccsd if spin not zero, else choose singlet ccsd
        if self.spin != 0:
            self.n_alpha = self.n_electrons//2 + self.spin//2 + 1 * (self.n_electrons % 2)
            self.n_beta = self.n_electrons//2 - self.spin//2
            self.n_singles, self.n_doubles, _, _, _, _, _ = uccsd_openshell_paramsize(self.n_spinorbitals, self.n_alpha, self.n_beta)
        else:
            self.n_spatial_orbitals = self.n_spinorbitals // 2
            self.n_occupied = int(np.ceil(self.n_electrons / 2))
            self.n_virtual = self.n_spatial_orbitals - self.n_occupied
            self.n_singles = self.n_occupied * self.n_virtual
            self.n_doubles = self.n_singles * (self.n_singles + 1) // 2

        # set total number of parameters
        self.n_var_params = self.n_singles + self.n_doubles

        # Supported reference state initialization
        # TODO: support for others
        self.supported_reference_state = {"HF"}
        # Supported var param initialization
        self.supported_initial_var_params = {"ones", "random"} if self.spin == 0 else {"ones", "random"}

        # Default initial parameters for initialization
        # TODO: support for openshell MP2 initialization
        self.var_params_default = "ones" if self.spin == 0 else "ones"
        self.default_reference_state = "HF"

        self.var_params = None
        self.circuit = None

    def set_var_params(self, var_params=None):
        """ Set values for variational parameters, such as zeros, random numbers, MP2 (...), providing some
        keywords for users, and also supporting direct user input (list or numpy array)
        Return the parameters so that workflows such as VQE can retrieve these values. """

        if var_params is None:
            var_params = self.var_params_default

        if isinstance(var_params, str):
            if (var_params not in self.supported_initial_var_params):
                raise ValueError(f"Supported keywords for initializing variational parameters: {self.supported_initial_var_params}")
            if var_params == "ones":
                initial_var_params = np.ones((self.n_var_params,), dtype=float)
            elif var_params == "random":
                initial_var_params = 2.e-1 * (np.random.random((self.n_var_params,)) - 0.5)
            elif var_params == "MP2":
                initial_var_params = self._compute_mp2_params()
        else:
            try:
                assert (len(var_params) == self.n_var_params)
                initial_var_params = np.array(var_params)
            except AssertionError:
                raise ValueError(f"Expected {self.n_var_params} variational parameters but received {len(var_params)}.")
        self.var_params = initial_var_params
        return initial_var_params

    def prepare_reference_state(self):
        """ Returns circuit preparing the reference state of the ansatz (e.g prepare reference wavefunction with HF,
        multi-reference state, etc). These preparations must be consistent with the transform used to obtain the
        qubit operator.
        """

        if self.default_reference_state not in self.supported_reference_state:
            raise ValueError(f"Only supported reference state methods are:{self.supported_reference_state}")

        if self.default_reference_state == "HF":
            return get_reference_circuit(n_spinorbitals=self.n_spinorbitals,
                                         n_electrons=self.n_electrons,
                                         mapping=self.mapping,
                                         up_then_down=self.up_then_down,
                                         spin=self.spin)

    def build_circuit(self, var_params=None):
        """ Build and return the quantum circuit implementing the state preparation ansatz
         (with currently specified initial_state and var_params) """

        if var_params is not None:
            self.set_var_params(var_params)
        elif self.var_params is None:
            self.set_var_params()

        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit_operator() if self.spin == 0 else self._get_openshell_qubit_operator()

        # Prepend reference state circuit
        reference_state_circuit = self.prepare_reference_state()

        # Obtain quantum circuit through trivial trotterization of the qubit operator
        # Keep track of the order in which pauli words have been visited for fast subsequent parameter updates
        pauli_words = sorted(qubit_op.terms.items(), key=lambda x: len(x[0]))
        pauli_words_gates = []
        self.pauli_to_angles_mapping = dict()
        for i, (pauli_word, coef) in enumerate(pauli_words):
            pauli_words_gates += pauliword_to_circuit(pauli_word, coef)
            self.pauli_to_angles_mapping[pauli_word] = i

        uccsd_circuit = Circuit(pauli_words_gates)
        # skip over the reference state circuit if it is empty
        if reference_state_circuit.size != 0:
            self.circuit = reference_state_circuit + uccsd_circuit
        else:
            self.circuit = uccsd_circuit

    def update_var_params(self, var_params):
        """ Shortcut: set value of variational parameters in the already-built ansatz circuit member.
            Preferable to rebuilt your circuit from scratch, which can be an involved process. """

        self.set_var_params(var_params)

        # Build qubit operator required to build UCCSD
        qubit_op = self._get_singlet_qubit_operator() if self.spin == 0 else self._get_openshell_qubit_operator()

        # If qubit operator terms have changed, rebuild circuit. Else, simply update variational gates directly
        if set(self.pauli_to_angles_mapping.keys()) != set(qubit_op.terms.keys()):
            self.build_circuit(var_params)
        else:
            for pauli_word, coef in qubit_op.terms.items():
                gate_index = self.pauli_to_angles_mapping[pauli_word]
                self.circuit._variational_gates[gate_index].parameter = 2.*coef if coef >= 0. else 4*np.pi+2*coef

    def _get_singlet_qubit_operator(self):
        """Construct UCCSD FermionOperator for current variational parameters, and translate to QubitOperator
        via relevant qubit mapping.

        Returns:
            qubit_op (QubitOperator): qubit-encoded elements of the UCCSD ansatz.
        """
        fermion_op = uccsd_singlet_generator(self.var_params, self.n_spinorbitals, self.n_electrons)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.mapping,
                                            n_spinorbitals=self.n_spinorbitals,
                                            n_electrons=self.n_electrons,
                                            up_then_down=self.up_then_down)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op

    def _get_openshell_qubit_operator(self):
        """Construct open-shell UCCSD FermionOperator for current variational parameters, and translate to QubitOperator
        via relevant qubit mapping.

        Returns:
            qubit_op (QubitOperator): qubit-encoded elements of the UCCSD ansatz.
        """
        fermion_op = uccsd_openshell_generator(self.var_params,
                                               self.n_spinorbitals,
                                               self.n_alpha,
                                               self.n_beta)
        qubit_op = fermion_to_qubit_mapping(fermion_operator=fermion_op,
                                            mapping=self.mapping,
                                            n_spinorbitals=self.n_spinorbitals,
                                            n_electrons=self.n_electrons,
                                            up_then_down=self.up_then_down)

        # Cast all coefs to floats (rotations angles are real)
        for key in qubit_op.terms:
            qubit_op.terms[key] = float(qubit_op.terms[key].imag)
        qubit_op.compress()
        return qubit_op
