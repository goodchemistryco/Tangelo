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

"""Unit tests for closed-shell and restricted open-shell qubit coupled cluster
with involutory linear combinations (ILC) of anti-commuting sets (ACS) of Pauli words."""

import unittest
import numpy as np

from tangelo.linq import Simulator
from tangelo.toolboxes.ansatz_generator.ilc import ILC
from tangelo.toolboxes.operators.operators import QubitOperator
from tangelo.molecule_library import mol_H2_sto3g, mol_H4_sto3g, mol_H4_cation_sto3g

sim = Simulator()


class ILCTest(unittest.TestCase):
    """Unit tests for various functionalities of the ILC ansatz class. Examples for both closed-
    and restricted open-shell ILC are provided using H2, H4, and H4 +.
    """

    @staticmethod
    def test_ilc_set_var_params():
        """ Verify behavior of set_var_params for different inputs (keyword, list, numpy array). """

        ilc_ansatz = ILC(mol_H2_sto3g, up_then_down=True)

        one_zero = np.zeros((1,), dtype=float)

        ilc_ansatz.set_var_params("qmf_state")
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, one_zero, decimal=6)

        ilc_ansatz.set_var_params([0.])
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, one_zero, decimal=6)

        one_tenth = 0.1 * np.ones((1,))

        ilc_ansatz.set_var_params([0.1])
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, one_tenth, decimal=6)

        ilc_ansatz.set_var_params(np.array([0.1]))
        np.testing.assert_array_almost_equal(ilc_ansatz.var_params, one_tenth, decimal=6)

    def test_ilc_incorrect_number_var_params(self):
        """ Return an error if user provide incorrect number of variational parameters """

        ilc_ansatz = ILC(mol_H2_sto3g, up_then_down=True)

        self.assertRaises(ValueError, ilc_ansatz.set_var_params, np.array([1.] * 2))

    def test_ilc_h2(self):
        """ Verify closed-shell functionality when using the ILC class separately for H2 """

        # Build the ILC ansatz, which sets the QMF parameters automatically if none are passed
        ilc_var_params = [0.11360304]
        ilc_op_list = [QubitOperator("X0 Y1 Y2 Y3")]
        ilc_ansatz = ILC(mol_H2_sto3g, up_then_down=True, ilc_op_list=ilc_op_list)

        # Build a QMF + ILC circuit
        ilc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = ilc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        ilc_ansatz.update_var_params(ilc_var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, ilc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.1372697, delta=1e-6)

    def test_ilc_h4(self):
        """ Verify restricted open-shell functionality when using the ILC class for H4 """

        # Build the ILC ansatz, which sets the QMF parameters automatically if none are passed
        ilc_op_list = [QubitOperator("Z0 X1 Z3 Y4 Z5"), QubitOperator("X0 X1 Y2 X3 Y4 Y5"), QubitOperator("Z0 X1 Y2 Y4 Y5"),
                       QubitOperator("Y0 X1 Y2 Y4"), QubitOperator("Y1 X3 Y4 Y5"), QubitOperator("Y0 Y1 X3 Y4 Z5"),
                       QubitOperator("X0 Y1 X2 Z3 Y4 Y5"), QubitOperator("X0 Y1 Z2 Z3 Y4 Y5"), QubitOperator("Z0 X1 Y2 Y3 Y4 Z5")]
        ilc_var_params = [ 0.01902128, -0.01425816,  0.68617146,  0.18603782, -0.20042697,  0.07894877,
                          -0.09087040,  0.06681858, -0.07056268]
        ilc_ansatz = ILC(mol_H4_sto3g, "SCBK", False, ilc_op_list)

        # Build a QMF + ILC circuit
        ilc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = ilc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        ilc_ansatz.update_var_params(ilc_var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, ilc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.9608801, delta=1e-6)

    def test_ilc_h4_cation(self):
        """ Verify restricted open-shell functionality when using the ILC class for H4 + """

        # Build the ILC ansatz, which sets the QMF parameters automatically if none are passed
        ilc_op_list = [QubitOperator("Y0 Z2 X4 Z6"), QubitOperator("Y1 Y2 Z4 X5 Y6"), QubitOperator("X0 Z2 Z4 Y6"),
                       QubitOperator("X1 Y2 X4 Z6"), QubitOperator("Y1 Y2 X4 Y5 Z6"), QubitOperator("Y1 Y2 Z4 Z5 Y6"),
                       QubitOperator("Y0 Z1 Z2 Y5 Y6"), QubitOperator("Y0 Z1 Z2 Y4 Y5 Z6")]
        ilc_var_params = [ 0.14017492, -0.10792805, -0.05835484,  0.12468933,  0.07173118,  0.04683807,  0.02852163, -0.03133538]
        ilc_ansatz = ILC(mol_H4_cation_sto3g, "BK", False, ilc_op_list)

        # Build a QMF + ILC circuit
        ilc_ansatz.build_circuit()

        # Get qubit hamiltonian for energy evaluation
        qubit_hamiltonian = ilc_ansatz.qubit_ham

        # Assert energy returned is as expected for given parameters
        ilc_ansatz.update_var_params(ilc_var_params)
        energy = sim.get_expectation_value(qubit_hamiltonian, ilc_ansatz.circuit)
        self.assertAlmostEqual(energy, -1.6379638, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
