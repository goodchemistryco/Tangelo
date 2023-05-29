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

from tangelo.algorithms.classical import MP2Solver
from tangelo.molecule_library import mol_H2_321g, mol_Be_321g, mol_H4_cation_sto3g


class MP2SolverTest(unittest.TestCase):

    def test_h2(self):
        """Test MP2Solver against result from reference implementation (H2)."""

        solver = MP2Solver(mol_H2_321g)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -1.14025452, places=6)

    def test_be(self):
        """Test MP2Solver against result from reference implementation (Be)."""

        solver = MP2Solver(mol_Be_321g)
        energy = solver.simulate()
        self.assertAlmostEqual(energy, -14.51026131, places=6)

        # Assert energy calculated from RDMs and MP2 calculation are the same.
        one_rdm, two_rdm = solver.get_rdm()
        self.assertAlmostEqual(mol_Be_321g.energy_from_rdms(one_rdm, two_rdm), energy)

    def test_get_rdm_without_simulate(self):
        """Test that the runtime error is raised when user calls get RDM without
        first running a simulation.
        """

        solver = MP2Solver(mol_H2_321g)
        self.assertRaises(RuntimeError, solver.get_rdm)

    def test_be_frozen_core(self):
        """ Test MP2Solver against result from reference implementation, with no mean-field provided as input.
            Frozen core is considered.
        """

        mol_Be_321g_freeze1 = mol_Be_321g.freeze_mos(1, inplace=False)

        solver = MP2Solver(mol_Be_321g_freeze1)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.5092873, places=6)

    def test_fci_be_frozen_core(self):
        """ Test FCISolver against result from reference implementation, with no mean-field provided as input.
            Frozen core is considered.
        """

        mol_Be_321g_freeze1 = mol_Be_321g.freeze_mos(1, inplace=False)

        solver = MP2Solver(mol_Be_321g_freeze1)
        energy = solver.simulate()

        self.assertAlmostEqual(energy, -14.5092873, places=6)


if __name__ == "__main__":
    unittest.main()
