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

import unittest

from numpy.linalg import eigh
from openfermion import get_sparse_operator

from tangelo.linq import Simulator, Circuit
from tangelo.molecule_library import mol_H4_sto3g
from tangelo.toolboxes.ansatz_generator.diagonal_coulomb import get_orbital_rotations
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping

# Initiate simulator using cirq as it has the same ordering as openfermion and we are using an exact eigenvector to test
sim = Simulator(target="cirq")


class diagonal_coulomb_Test(unittest.TestCase):

    def test_orbital_rotations(self):
        """test calculating energy expectation value of H4 hamiltonian by decomposing into diagional terms"""

        # Generate ground state wavefunction
        ham = get_sparse_operator(mol_H4_sto3g.fermionic_hamiltonian).toarray()
        eigs, vecs = eigh(ham)
        state_vec = vecs[:, 0]

        # Generate necessary circuits and operators
        orb_rots = get_orbital_rotations(mol_H4_sto3g)

        # Run each set of gates that diagonalizes the set of terms and calculate energy
        energy = 0
        for i in range(len(orb_rots.constants)):
            qu_op = fermion_to_qubit_mapping(orb_rots.fermion_operators[i], "JW")
            freqs, _ = sim.simulate(Circuit(orb_rots.rotation_gates[i]), initial_statevector=state_vec)
            for term, coeff in qu_op.terms.items():
                energy += coeff*sim.get_expectation_value_from_frequencies_oneterm(term, frequencies=freqs)

        self.assertAlmostEqual(energy, eigs[0], places=6)


if __name__ == "__main__":
    unittest.main()
