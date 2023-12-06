# Copyright 2023 Good Chemsitry Company.
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

import unittest

import numpy as np
from openfermion import get_sparse_operator

from tangelo.algorithms.projective.qpe import QPESolver
from tangelo.algorithms.projective.iqpe import IterativeQPESolver
from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize
from tangelo.toolboxes.operators import QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_reference_circuit
from tangelo.molecule_library import mol_H2_sto3g
from tangelo.linq.helpers.circuits.statevector import StateVector


class QPESolverTest(unittest.TestCase):

    def test_instantiation(self):
        """Try instantiating QPESolver with basic input."""

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw"}
        IterativeQPESolver(options)

    def test_instantiation_incorrect_keyword(self):
        """Instantiating with an incorrect keyword should return an error """

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw", "dummy": True}
        self.assertRaises(KeyError, IterativeQPESolver, options)

    def test_instantiation_missing_molecule(self):
        """Instantiating with no molecule should return an error."""

        options = {"qubit_mapping": "jw"}
        self.assertRaises(ValueError, IterativeQPESolver, options)

    def test_simulate_h2(self):
        """Run QPE on H2 molecule, with scbk qubit mapping and exact simulator with the approximate initial state
        """

        qpe_options = {"molecule": mol_H2_sto3g, "qubit_mapping": "scbk", "up_then_down": True, "size_qpe_register": 7,
                       "backend_options": {"target": "qulacs", "n_shots": 20}, "unitary_options": {"time": 2*np.pi, "n_trotter_steps": 1,
                                                                                    "n_steps_method": "repeat", "trotter_order": 4}}
        qpe_solver = IterativeQPESolver(qpe_options)
        qpe_solver.build()

        _ = qpe_solver.simulate()
        # Use the highest probability circuit which is about 0.5. Will fail ~1 in every 2^20 times.
        max_prob_key = max(qpe_solver.circuit.success_probabilities, key=qpe_solver.circuit.success_probabilities.get)
        self.assertAlmostEqual(qpe_solver.energy_estimation(max_prob_key[::-1]), 0.14, delta=1e-2)

        # Test that get_resources returns expected results
        resources = qpe_solver.get_resources()
        self.assertEqual(resources["qubit_hamiltonian_terms"], 5)
        self.assertEqual(resources["circuit_width"], 3)

    def test_simulate_h2_circuit(self):
        """Run QPE on H2 molecule, with scbk qubit mapping and exact simulator providing only the Trotter circuit and
        the exact initial state.
        """

        qu_op = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, "scbk", mol_H2_sto3g.n_active_sos,
                                         mol_H2_sto3g.n_active_electrons, True, 0)
        ham_mat = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, wavefunction = np.linalg.eigh(ham_mat)

        sv = StateVector(wavefunction[:, 0], order="lsq_first")
        ref_circ = sv.initializing_circuit()
        unit_circ = trotterize(mol_H2_sto3g.fermionic_hamiltonian, 2*np.pi, 1, 4, True,
                               {"qubit_mapping": "scbk", "up_then_down": True, "n_spinorbitals": mol_H2_sto3g.n_active_sos,
                                "n_electrons": mol_H2_sto3g.n_active_electrons})

        # Test supplying circuit and applying QPE controls to only gates marked as variational
        qpe_options = {"unitary": unit_circ, "size_qpe_register": 7, "ref_state": ref_circ,
                       "backend_options": {"target": "qulacs", "n_shots": 1}, "unitary_options": {"control_method": "variational"}}
        qpe_solver = IterativeQPESolver(qpe_options)
        qpe_solver.build()

        energy = qpe_solver.simulate()

        self.assertAlmostEqual(energy, -(-1.13727-qu_op.constant), delta=1e-3)

        # Test supplying circuit with QPE controls added to every gate.
        qpe_options = {"unitary": unit_circ, "size_qpe_register": 7, "ref_state": ref_circ,
                       "backend_options": {"target": "qulacs", "n_shots": 1}, "unitary_options": {"control_method": "all"}}
        qpe_solver = IterativeQPESolver(qpe_options)
        qpe_solver.build()

        energy = qpe_solver.simulate()

        self.assertAlmostEqual(energy, -(-1.13727-qu_op.constant), delta=1e-3)

    def test_qubit_hamiltonian_input(self):
        """Test with qubit hamiltonian input."""

        # Generate qubit operator with state 9 having eigenvalue 0.25
        qu_op = (QubitOperator("X0 X1", 0.125) + QubitOperator("Y1 Y2", 0.125) + QubitOperator("Z2 Z3", 0.125)
                 + QubitOperator("", 0.125))

        ham_mat = get_sparse_operator(qu_op.to_openfermion()).toarray()
        _, wavefunction = np.linalg.eigh(ham_mat)

        sv = StateVector(wavefunction[:, 9], order="lsq_first")
        init_circ = sv.initializing_circuit()

        qpe = IterativeQPESolver({"qubit_hamiltonian": qu_op, "size_qpe_register": 6, "ref_state": init_circ,
                                  "backend_options": {"noise_model": None, "target": "cirq"},
                                  "unitary_options": {"time": -2*np.pi, "n_trotter_steps": 1,
                                                      "n_steps_method": "repeat", "trotter_order": 4}})
        qpe.build()
        energy = qpe.simulate()
        self.assertAlmostEqual(energy, 0.25, delta=1.e-5)


if __name__ == "__main__":
    unittest.main()
