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

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Module to create and manipulate unitary coupled cluster operators for
   open shell systems. This requires the number of alpha and beta electrons
   to be specified.
"""

import itertools
from typing import Dict, Tuple

import numpy
from openfermion.utils import down_index, up_index

from tangelo.toolboxes.operators import FermionOperator


def uccsd_singlet_generator(n_qubits,
                            n_electrons,
                            anti_hermitian=True) -> Tuple[Dict[int, Tuple[int]], Dict[Tuple[int], FermionOperator]]:
    """Create two dictionaries that map the packed amplitudes to the corresponding UCCSD generator.

    This function generates the FermionOperator for each UCCSD generator designed
        to act on a single reference state consisting of n_qubits spin orbitals
        and n_electrons electrons, that is a spin singlet operator, meaning it
        conserves spin.

    Args:
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.
        anti_hermitian(Bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing

    Returns:
        Tuple[Dict, Dict]: (Mapping from packed_amplitude integer to generator, Mapping from generator to FermionOperator)
    """
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Get amplitude indices
    n_single_amplitudes = n_occupied * n_virtual

    # Parameter dict
    param_dict = dict()

    # Operator dict
    operator_dict = dict()

    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    for i, (p, q) in enumerate(
            itertools.product(range(n_virtual), range(n_occupied))):

        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q

        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            this_index = spin_index_functions[spin]
            other_index = spin_index_functions[1 - spin]

            # Get indices of spin orbitals
            virtual_this = this_index(virtual_spatial)
            virtual_other = other_index(virtual_spatial)
            occupied_this = this_index(occupied_spatial)
            occupied_other = other_index(occupied_spatial)

            # Generate single excitations
            generator = FermionOperator(
                ((virtual_this, 1), (occupied_this, 0)), 1)
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_this, 1), (virtual_this, 0)), -1)
            excitation = (virtual_spatial, occupied_spatial)
            param_dict[i] = excitation
            operator_dict[excitation] = operator_dict.get(excitation, FermionOperator()) + generator

            # Generate double excitation
            generator = FermionOperator(
                ((virtual_this, 1), (occupied_this, 0), (virtual_other, 1),
                 (occupied_other, 0)), 1)
            if anti_hermitian:
                generator += FermionOperator(
                    ((occupied_other, 1), (virtual_other, 0),
                     (occupied_this, 1), (virtual_this, 0)), -1)
            excitation = (virtual_spatial, occupied_spatial, virtual_spatial, occupied_spatial)
            param_dict[i + n_single_amplitudes] = excitation
            operator_dict[excitation] = operator_dict.get(excitation, FermionOperator()) + generator

    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_virtual), range(n_occupied)), 2)):

        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s

        # Generate double excitations
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            index_a = spin_index_functions[spin_a]
            index_b = spin_index_functions[spin_b]

            # Get indices of spin orbitals
            virtual_1_a = index_a(virtual_spatial_1)
            occupied_1_a = index_a(occupied_spatial_1)
            virtual_2_b = index_b(virtual_spatial_2)
            occupied_2_b = index_b(occupied_spatial_2)

            if virtual_1_a == virtual_2_b:
                continue
            if occupied_1_a == occupied_2_b:
                continue
            else:

                generator = FermionOperator(
                    ((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1),
                     (occupied_2_b, 0)), 1)
                if anti_hermitian:
                    generator += FermionOperator(
                        ((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1),
                         (virtual_1_a, 0)), -1)
                excitation = (virtual_spatial_1, occupied_spatial_1, virtual_spatial_2, occupied_spatial_2)
                param_dict[i + 2*n_single_amplitudes] = excitation
                operator_dict[excitation] = operator_dict.get(excitation, FermionOperator()) + generator

    return param_dict, operator_dict
