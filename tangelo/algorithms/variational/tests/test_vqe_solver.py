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

import unittest
import numpy as np

from tangelo.linq import get_backend, Circuit, Gate
from tangelo.helpers.utils import installed_backends
from tangelo.linq.target import QiskitSimulator
from tangelo.algorithms import BuiltInAnsatze, VQESolver
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g, mol_NaH_sto3g, mol_H4_sto3g_symm, mol_H4_sto3g_uhf_a1_frozen
from tangelo.toolboxes.ansatz_generator.uccsd import UCCSD
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.molecular_computation.rdms import matricize_2rdm
from tangelo.toolboxes.optimizers.rotosolve import rotosolve
from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule
from tangelo.toolboxes.ansatz_generator.fermionic_operators import spinz_operator
from tangelo.toolboxes.ansatz_generator.ansatz_utils import trotterize, get_qft_circuit


class VQESolverTest(unittest.TestCase):

    def test_instantiation_vqe(self):
        """Try instantiating VQESolver with basic input."""

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw"}
        VQESolver(options)

    def test_instantiation_vqe_incorrect_keyword(self):
        """Instantiating with an incorrect keyword should return an error """

        options = {"molecule": mol_H2_sto3g, "qubit_mapping": "jw", "dummy": True}
        self.assertRaises(KeyError, VQESolver, options)

    def test_instantiation_vqe_missing_molecule(self):
        """Instantiating with no molecule should return an error."""

        options = {"qubit_mapping": "jw"}
        self.assertRaises(ValueError, VQESolver, options)

    def test_get_resources_h2_mappings(self):
        """Resource estimation, with UCCSD ansatz, given initial parameters.
        Each of JW, BK, and scBK mappings are checked.
        """
        mappings = ["jw", "bk", "scbk", "jkmn"]
        expected_values = [(15, 4), (15, 4), (5, 2), (15, 4)]

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": [0.1, 0.1]}
        for index, mi in enumerate(mappings):
            vqe_options["qubit_mapping"] = mi
            vqe_solver = VQESolver(vqe_options)
            vqe_solver.build()
            resources = vqe_solver.get_resources()

            self.assertEqual(resources["qubit_hamiltonian_terms"], expected_values[index][0])
            self.assertEqual(resources["circuit_width"], expected_values[index][1])

    def test_energy_estimation_vqe(self):
        """A single VQE energy evaluation for H2, using optimal parameters and
        exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw"}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.energy_estimation([5.86665842e-06, 5.65317429e-02])
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

    def test_operator_expectation_vqe(self):
        """ A test of the operator_expectation function, using optimal parameters and exact simulator """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": 'jw'}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Test using var_params input and Qubit Hamiltonian
        energy = vqe_solver.operator_expectation(vqe_solver.qubit_hamiltonian, var_params=[5.86665842e-06, 5.65317429e-02])
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

        # Test using updated var_params and Fermion Hamiltonian
        vqe_solver.ansatz.update_var_params([5.86665842e-06, 5.65317429e-02])
        energy = vqe_solver.operator_expectation(mol_H2_sto3g.fermionic_hamiltonian)
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

        # Test the three in place operators
        n_electrons = vqe_solver.operator_expectation('N')
        self.assertAlmostEqual(n_electrons, 2, places=6)
        spin_z = vqe_solver.operator_expectation('Sz')
        self.assertAlmostEqual(spin_z, 0, places=6)
        spin2 = vqe_solver.operator_expectation('S^2')
        self.assertAlmostEqual(spin2, 0, places=6)

    def test_simulate_h2(self):
        """Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": [0.1, 0.1], "verbose": True}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_simulate_h2_with_deflation(self):
        """Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator. Followed by UpCCGSD with deflation of ground state.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": [0.1, 0.1], "verbose": True}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

        # Use deflation to get first excited state.
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UpCCGSD, "qubit_mapping": "jw",
                       "verbose": True, "deflation_circuits": [vqe_solver.optimal_circuit],
                       "deflation_coeff": 1.0, "ref_state": [0, 1, 0, 1]}
        vqe_solver_2 = VQESolver(vqe_options)
        vqe_solver_2.build()

        energy = vqe_solver_2.simulate()
        self.assertAlmostEqual(energy, -0.53247, delta=1e-4)

    def test_simulate_qmf_h2(self):
        """Run VQE on H2 molecule, with QMF ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.QMF, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.116684, delta=1e-4)

    def test_simulate_qcc_h2(self):
        """Run VQE on H2 molecule, with QCC ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.QCC, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270, delta=1e-4)

    def test_simulate_ilc_h2(self):
        """Run VQE on H2 molecule, with ILC ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.ILC, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270, delta=1e-4)

    def test_simulate_vsqs_h2(self):
        """Run VQE on H2 molecule, with vsqs ansatz, JW qubit mapping, exact simulator for both molecule input and
        qubit_hamiltonian/hini/reference_state input
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.VSQS, "qubit_mapping": "jw",
                        "verbose": False, "ansatz_options": {"intervals": 3, "time": 3}}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270, delta=1e-4)

        qubit_hamiltonian = vqe_solver.qubit_hamiltonian
        h_init = vqe_solver.ansatz.h_init
        reference_state = vqe_solver.ansatz.prepare_reference_state()

        vqe_options = {"molecule": None, "qubit_hamiltonian": qubit_hamiltonian, "ansatz": BuiltInAnsatze.VSQS, "qubit_mapping": "jw",
                        "ansatz_options": {"intervals": 3, "time": 3, "qubit_hamiltonian": qubit_hamiltonian,
                                          "h_init": h_init, "reference_state": reference_state}}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270, delta=1e-4)

    @unittest.skipIf("qiskit" not in installed_backends, "Test Skipped: Backend not available \n")
    def test_simulate_h2_qiskit(self):
        """Run VQE on H2 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact qiskit simulator. Both string and class input.
        """

        backend_options = {"target": "qiskit", "n_shots": None, "noise_model": None}
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": [6.28531447e-06, 5.65431626e-02], "verbose": True,
                        "backend_options": backend_options}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        self.assertAlmostEqual(energy, -1.13727042117, delta=1e-6)

        backend_options = {"target": QiskitSimulator, "n_shots": None, "noise_model": None}
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": [6.28531447e-06, 5.65431626e-02], "verbose": True,
                        "backend_options": backend_options}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        self.assertAlmostEqual(energy, -1.13727042117, delta=1e-6)

    def test_simulate_h4(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """
        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.9778312978826869, delta=1e-4)

    def test_simulate_h4_symm(self):
        """Run VQE on H2 molecule with symmetry turned on, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H4_sto3g_symm, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                       "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.97783, delta=1e-4)
        # Test that the number of variational parameters is less than the total possible number
        self.assertEqual(vqe_solver.ansatz.n_var_params, 10)
        self.assertEqual(vqe_solver.ansatz.n_full_var_params, 14)

    def test_simulate_qmf_h4(self):
        """Run VQE on H4 molecule, with QMF ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.QMF, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.789483, delta=1e-4)

    def test_simulate_qcc_h4(self):
        """Run VQE on H4 molecule, with QCC ansatz, JW qubit mapping, initial
        parameters, exact simulator. Followed by calculation with Sz symmetry projection
        """

        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.QCC, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.963270, delta=1e-4)

        # QPE based Sz projection.
        def sz_check(n_state: int, molecule: SecondQuantizedMolecule, mapping: str, up_then_down):
            n_qft = 3
            spin_fe_op = spinz_operator(molecule.n_active_mos)
            q_spin = fermion_to_qubit_mapping(spin_fe_op, mapping, molecule.n_active_sos, molecule.n_active_electrons, up_then_down, molecule.spin)

            sym_var_circuit = Circuit([Gate("H", q) for q in range(n_state, n_state+n_qft)])
            for j, i in enumerate(range(n_state, n_state+n_qft)):
                sym_var_circuit += trotterize(2*q_spin+3, -2*np.pi/2**(j+1), control=i)
            sym_var_circuit += get_qft_circuit(list(range(n_state+n_qft-1, n_state-1, -1)), inverse=True)
            sym_var_circuit += Circuit([Gate("MEASURE", i) for i in range(n_state, n_state+n_qft)])
            return sym_var_circuit

        # Use a circuit with variational gates and mid-circuit measurements as the ansatz.
        proj_circuit = sz_check(8, mol_H4_sto3g, "JW", vqe_solver.up_then_down)
        var_circuit = vqe_solver.optimal_circuit + proj_circuit

        vqe_solver_p = VQESolver({"ansatz": var_circuit, "qubit_hamiltonian": vqe_solver.qubit_hamiltonian, "simulate_options": {"desired_meas_result": "011"}})
        vqe_solver_p.build()
        energyp = vqe_solver_p.simulate()
        self.assertAlmostEqual(energyp, -1.97622, delta=1e-4)

        # Use a circuit with variational gates as the ansatz, add a projective circuit separately.
        var_circuit = vqe_solver.optimal_circuit
        vqe_solver_p = VQESolver({"ansatz": var_circuit, "qubit_hamiltonian": vqe_solver.qubit_hamiltonian,
                                  "simulate_options": {"desired_meas_result": "011"}, "projective_circuit": proj_circuit})
        vqe_solver_p.build()
        energyp = vqe_solver_p.simulate()
        self.assertAlmostEqual(energyp, -1.97622, delta=1e-4)

    def test_simulate_ilc_h4(self):
        """Run VQE on H4 molecule, with ILC ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.ILC, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.960877, delta=1e-4)

    def test_simulate_h4_open(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial parameters, exact simulator """
        vqe_options = {"molecule": mol_H4_cation_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": "random", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.6394, delta=1e-3)

    def test_simulate_h4_uhf_a1_frozen(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, scbk qubit mapping, initial parameters, exact simulator """
        vqe_options = {"molecule": mol_H4_sto3g_uhf_a1_frozen, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "scbk",
                        "initial_var_params": [0.001]*15, "verbose": False, "up_then_down": True}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.95831, delta=1e-3)

    def test_simulate_qmf_h4_open(self):
        """Run VQE on H4 + molecule, with QMF ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H4_cation_sto3g, "ansatz": BuiltInAnsatze.QMF, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.585918, delta=1e-4)

    def test_simulate_qcc_h4_open(self):
        """Run VQE on H4 + molecule, with QCC ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H4_cation_sto3g, "ansatz": BuiltInAnsatze.QCC, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.638020, delta=1e-4)

    def test_simulate_ilc_h4_open(self):
        """Run VQE on H4 + molecule, with ILC ansatz, JW qubit mapping, initial
        parameters, exact simulator.
        """

        vqe_options = {"molecule": mol_H4_cation_sto3g, "ansatz": BuiltInAnsatze.ILC, "qubit_mapping": "jw",
                        "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.638020, delta=1e-4)

    def test_optimal_circuit_h4(self):
        """Run VQE on H4 molecule, save optimal circuit. Verify it yields
        optimal energy.
        """
        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        sim = get_backend()
        self.assertAlmostEqual(energy, sim.get_expectation_value(vqe_solver.qubit_hamiltonian, vqe_solver.optimal_circuit),
                                delta=1e-10)

    def test_get_rdm_h2(self):
        """Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized
        parameters, exact simulator (H2).
        """

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw"}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Compute RDM matrices
        one_rdm, two_rdm = vqe_solver.get_rdm([5.86665842e-06, 5.65317429e-02])

        # Test traces of matrices
        n_elec, n_orb = mol_H2_sto3g.n_active_electrons, mol_H2_sto3g.n_active_mos
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                                delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                                msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_get_rdm_h4(self):
        """Compute RDMs with UCCSD ansatz, JW qubit mapping, optimized
        parameters, exact simulator (H4).
        """

        vqe_options = {"molecule": mol_H4_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw"}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        # Compute RDM matrices
        var_params = [-6.47627367e-06, -5.24257363e-06, -5.99540594e-06, -7.70205325e-06, 1.15628926e-02,
                      3.42313563e-01,  3.48211343e-02,  1.49150233e-02, 7.53406401e-02,  8.44095525e-03,
                      -1.79981377e-01, -1.00585201e-01, 1.02162534e-02, -3.65870070e-02]
        one_rdm, two_rdm = vqe_solver.get_rdm(var_params)
        # Test traces of matrices
        n_elec, n_orb = mol_H4_sto3g.n_active_electrons, mol_H4_sto3g.n_active_mos
        self.assertAlmostEqual(np.trace(one_rdm), n_elec, msg="Trace of one_rdm does not match number of electrons",
                                delta=1e-6)
        rho = matricize_2rdm(two_rdm, n_orb)
        self.assertAlmostEqual(np.trace(rho), n_elec * (n_elec - 1),
                                msg="Trace of two_rdm does not match n_elec * (n_elec-1)", delta=1e-6)

    def test_custom_vqe(self):
        """VQE with custom optimizer and non-optimal variational parameters."""

        # Define and assign custom optimizer: cobyla
        def cobyla_oneshot_optimizer(func, var_params):
            from scipy.optimize import minimize
            result = minimize(func, var_params, method="COBYLA", options={"disp": True, "maxiter": 100})
            return result.fun, result.x

        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": "ones", "verbose": False,
                        "optimizer": cobyla_oneshot_optimizer}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.137270422018, places=6)

    def test_mapping_BK(self):
        """Test that BK mapping recovers the expected result, to within 1e-6 Ha,
        for the example of H2 and MP2 initial guess.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
                        "qubit_mapping": "bk"}

        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_mapping_scBK(self):
        """Test that scBK mapping recovers the expected result, to within
        1e-6 Ha, for the example of H2 and MP2 initial guess.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
                        "qubit_mapping": "scbk"}

        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_mapping_jkmn(self):
        """Test that JKMN mapping recovers the expected result, to within
        1e-6 Ha, for the example of H2 and MP2 initial guess.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "verbose": False,
                        "qubit_mapping": "jkmn"}

        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()
        energy = vqe_solver.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy, energy_target, places=5)

    def test_spin_reorder_equivalence(self):
        """Test that re-ordered spin input (all up followed by all down) returns
        the same optimized energy result for both JW and BK mappings.
        """
        vqe_options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD, "initial_var_params": "MP2", "up_then_down": True,
                        "verbose": False, "qubit_mapping": "jw"}

        vqe_solver_jw = VQESolver(vqe_options)
        vqe_solver_jw.build()
        energy_jw = vqe_solver_jw.simulate()

        vqe_options["qubit_mapping"] = "bk"
        vqe_solver_bk = VQESolver(vqe_options)
        vqe_solver_bk.build()
        energy_bk = vqe_solver_bk.simulate()

        energy_target = -1.137270
        self.assertAlmostEqual(energy_jw, energy_target, places=5)
        self.assertAlmostEqual(energy_bk, energy_target, places=5)

    def test_simulate_h4_frozen_orbitals(self):
        """Run VQE on H4 molecule, with UCCSD ansatz, JW qubit mapping, initial
        parameters, exact simulator. First (occupied) and last (virtual)
        orbitals are frozen.
        """
        mol_H4_sto3g_frozen = mol_H4_sto3g.freeze_mos([0, 3], inplace=False)

        vqe_options = {"molecule": mol_H4_sto3g_frozen, "ansatz": BuiltInAnsatze.UCCSD, "qubit_mapping": "jw",
                        "initial_var_params": "MP2", "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        energy = vqe_solver.simulate()
        self.assertAlmostEqual(energy, -1.8943598012229799, delta=1e-5)

    def test_simulate_nah_rucc(self):
        """Run VQE on NaH molecule, with UCC1 and UCC3 ansatze, JW qubit
        mapping. The computation is mapped to a HOMO-LUMO problem.
        """

        mol_NaH_sto3g_2mos = mol_NaH_sto3g.freeze_mos([i for i in range(9) if i not in [5, 9]], inplace=False)

        vqe_options = {"molecule": mol_NaH_sto3g_2mos, "ansatz": BuiltInAnsatze.UCC1, "qubit_mapping": "jw",
                        "initial_var_params": "zeros", "up_then_down": True, "verbose": False}

        vqe_solver_ucc1 = VQESolver(vqe_options)
        vqe_solver_ucc1.build()
        energy_ucc1 = vqe_solver_ucc1.simulate()

        vqe_options["ansatz"] = BuiltInAnsatze.UCC3
        vqe_solver_ucc3 = VQESolver(vqe_options)
        vqe_solver_ucc3.build()
        energy_ucc3 = vqe_solver_ucc3.simulate()

        self.assertAlmostEqual(energy_ucc1, -160.30334365109297, delta=1e-6)
        self.assertAlmostEqual(energy_ucc3, -160.30345935884606, delta=1e-6)

    def test_toomany_orbitals_rucc(self):
        """Test the case where there is too many orbitals in the system to be
        mapped into a HOMO-LUMO problem.
        """

        vqe_options = {"molecule": mol_NaH_sto3g, "ansatz": BuiltInAnsatze.UCC1, "qubit_mapping": "jw",
                        "initial_var_params": "zeros", "up_then_down": True, "verbose": False}

        with self.assertRaises(ValueError):
            vqe_solver_ucc1 = VQESolver(vqe_options)
            vqe_solver_ucc1.build()

        with self.assertRaises(ValueError):
            vqe_options["ansatz"] = BuiltInAnsatze.UCC3
            vqe_solver_ucc3 = VQESolver(vqe_options)
            vqe_solver_ucc3.build()

    def test_wrong_mapping_rucc(self):
        """Test the case where another mapping process (not JW) is selected."""

        mol_NaH_sto3g_2mos = mol_NaH_sto3g.freeze_mos([i for i in range(9) if i not in [5, 9]], inplace=False)

        vqe_options = {"molecule": mol_NaH_sto3g_2mos, "ansatz": BuiltInAnsatze.UCC1, "qubit_mapping": "bk",
                        "initial_var_params": "zeros", "up_then_down": True}

        with self.assertRaises(ValueError):
            vqe_solver_ucc1 = VQESolver(vqe_options)
            vqe_solver_ucc1.build()

        with self.assertRaises(ValueError):
            vqe_options["ansatz"] = BuiltInAnsatze.UCC3
            vqe_solver_ucc3 = VQESolver(vqe_options)
            vqe_solver_ucc3.build()

    def test_qubit_qhamiltonian_input(self):
        """Test the case where a qubit Hamiltonian is used to construct VQE."""

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"qubit_hamiltonian": qubit_hamiltonian,
                    "ansatz": UCCSD(mol_H2_sto3g, mapping="jw")}
        VQESolver(options)

    def test_qubit_qhamiltonian_input_conflicts(self):
        """Test the case where a molecule and a qubit Hamiltonian are passed as
        inputs.
        """

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"molecule": mol_H2_sto3g,
                    "qubit_hamiltonian": qubit_hamiltonian,
                    "ansatz": UCCSD(mol_H2_sto3g, mapping="jw")}

        with self.assertRaises(ValueError):
            VQESolver(options)

    def test_qubit_qhamiltonian_input_no_custom_ansatz(self):
        """Test the case where no custom ansatz is passed when using a qubit
        Hamiltonian as input.
        """

        qubit_hamiltonian = fermion_to_qubit_mapping(mol_H2_sto3g.fermionic_hamiltonian, mapping="jw")

        options = {"qubit_hamiltonian": qubit_hamiltonian}

        with self.assertRaises(TypeError):
            VQESolver(options).build()

        options["ansatz"] = BuiltInAnsatze.UCCSD

        with self.assertRaises(TypeError):
            VQESolver(options).build()

    def test_rotosolve(self):
        """Run VQE on H2 molecule, using Rotosolve, with UCC3 ansatz,
        JW qubit mapping, and exact simulator.
        """

        options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCC3,
                   "qubit_mapping": "jw", "verbose": False,
                   "optimizer": rotosolve, "up_then_down": True}

        vqe_solver = VQESolver(options)
        vqe_solver.build()
        vqe_solver.simulate()
        energy = vqe_solver.optimal_energy

        self.assertAlmostEqual(energy, -1.137270422018, delta=1e-4)

    def test_rotosolve_unsupported(self):
        """Test Rotosolve unsupported Ansatz with UCCSD on H2 molecule with
        JW qubit mapping, and exact simulator.
        """

        options = {"molecule": mol_H2_sto3g, "ansatz": BuiltInAnsatze.UCCSD,
                   "qubit_mapping": "jw", "optimizer": rotosolve}

        with self.assertRaises(ValueError):
            VQESolver(options).build()

    def test_save_energies(self):
        """Performing a deterministic number of optimization steps (calls to
        energy_estimation). The energies attributes should have n elements,
        where n is the number of optimization steps.
        """

        vqe_options = {"molecule": mol_H2_sto3g,
                       "ansatz": Circuit([Gate("X", 0), Gate("X", 1)], n_qubits=4),
                       "qubit_mapping": "JW",
                       "up_then_down": False,
                       "save_energies": True,
                       "verbose": False}
        vqe_solver = VQESolver(vqe_options)
        vqe_solver.build()

        n_steps = 3
        # Deterministic number of calls to energy_estimation.
        for _ in range(n_steps):
            vqe_solver.energy_estimation(var_params=[])

        self.assertEqual(len(vqe_solver.energies), n_steps)


if __name__ == "__main__":
    unittest.main()
