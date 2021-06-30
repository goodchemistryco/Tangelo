"""This module provides a common point for mapping operators and
 statevectors from Fermionic to Qubit encodings via any of:
 - Jordan-Wigner
 - Bravyi-Kitaev (Fenwick Tree implementation)
 - symmetry-conserving Bravyi-Kitaev (2-qubit reduction via Z2 taper)
"""
import numpy as np
from collections.abc import Iterable

from qsdk.toolboxes.operators import FermionOperator
from qsdk.toolboxes.qubit_mappings import jordan_wigner, bravyi_kitaev, symmetry_conserving_bravyi_kitaev

available_mappings = {'JW', 'BK', 'SCBK'}


def get_qubit_number(mapping, n_spinorbitals):
    """Get the number of qubits for a specified number of spin orbitals,
    after passing through the qubit mapping. Symmetry-conserving
    Bravyi-Kitaev reduces the qubit number by 2.

    Args:
        mapping (string): qubit-mapping, JW, BK or SCBK
        n_spinorbitals (int): number of spin-orbitals for molecule

    Returns:
        (int): number of qubits
    """
    if mapping.upper() == 'SCBK':
        return n_spinorbitals - 2
    else:
        return n_spinorbitals


def get_fermion_operator(operator):
    """Cast operator to FermionOperator datatype. Input is of
    SymbolicOperator type, but term words must be valid input
    for FermionOperator, as for example for InteractionOperator.

    Args:
        operator (SymbolicOperator): input operator to be cast

    Returns:
        fermion_operator (FermionOperator)
    """
    if not isinstance(operator, Iterable):
        raise TypeError('Input must be iterable suitable for casting to FermionOperator type.')
    fermion_operator = FermionOperator()
    for term in operator:
        try:
            fermion_operator += FermionOperator(term, operator[term])
        except:
            raise TypeError('Operator terms are not formatted as valid input for FermionOperator type.')
    return fermion_operator


def fermion_to_qubit_mapping(fermion_operator, mapping, n_spinorbitals=None, n_electrons=None, up_then_down=False):
    """Perform mapping of fermionic operator to qubit operator. This function is mostly a wrapper
    around standard openfermion code, with some important distinctions. We strictly enforce the
    specification of n_qubits for Bravyi-Kitaev type transformations, and n_electrons for scBK.
    In the absence of this information, these methods can return unexpected results if the input
    operator does not specifically address the highest-orbital/qubit index.

    Args:
        fermion_operator (FermionOperator): operator to translate to qubit representation
        mapping (string): options are--
           'JW' (Jordan Wigner), 'BK' (Bravyi Kitaev), 'scBK' (symmetry-conserving Bravyi Kitaev).
        n_spinorbitals (int): number of spin-orbitals for problem. Not required for Jordan-Wigner
        n_electrons (int): number of occupied electron modes in problem. Required for symmetry
            conserving Bravyi-Kitaev only.
        up_then_down (bool): flag to change basis ordering, putting all spin up then all spin down.

    Returns:
        qubit_operator (QubitOperator): input operator, encoded in the qubit space.
    """
    # some methods may pass another operator class type. If this is the case, cast to FermionOperator where possible
    if not isinstance(fermion_operator, FermionOperator):
        fermion_operator = get_fermion_operator(fermion_operator)

    if mapping.upper() not in available_mappings:
        raise ValueError(f'Invalid mapping selection. Select from: {available_mappings}')

    if up_then_down:
        if n_spinorbitals is None:
            raise ValueError('The number of spin-orbitals is required to execute basis re-ordering.')
        fermion_operator = make_up_then_down(fermion_operator, n_spinorbitals=n_spinorbitals)

    if mapping.upper() == 'JW':
        qubit_operator = jordan_wigner(fermion_operator)
    elif mapping.upper() == 'BK':
        if n_spinorbitals is None:
            raise ValueError("Bravyi Kitaev requires specification of number of qubits.")
        qubit_operator = bravyi_kitaev(fermion_operator, n_qubits=n_spinorbitals)
    elif mapping.upper() == 'SCBK':
        if n_spinorbitals is None:
            raise ValueError("Symmetry-conserving Bravyi Kitaev requires specification of number of qubits.")
        if n_electrons is None:
            raise ValueError("Symmetry-conserving Bravyi Kitaev requires specification of number of electrons.")

        qubit_operator = symmetry_conserving_bravyi_kitaev(fermion_operator=fermion_operator,
                                                           n_spinorbitals=n_spinorbitals,
                                                           n_electrons=n_electrons,
                                                           up_then_down=up_then_down)

    return qubit_operator


def make_up_then_down(fermion_operator, n_spinorbitals):
    """Re-order the orbital indexing to force all spin up, followed by all spin down.
    The default ordering alternates between spin up and down.

    Args:
        fermion_operator (FermionOperator): input operator
        n_spinorbitals (int): number of spin-orbitals in register.

    Returns:
        new_operator (FermionOperator): operator with all spin up followed by all spin down.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError('Invalid operator input. Must be FermionOperator.')
    if n_spinorbitals % 2 != 0:
        raise ValueError('Invalid number of spin-orbitals. Expecting even number.')
    term_modes = max([factor[0] for term in fermion_operator.terms for factor in term]) + 1
    if term_modes > n_spinorbitals:
        raise ValueError('Invalid number of modes (n_spinorbitals) for input operator. Terms in operator exceed this value.')
    new_operator = FermionOperator()

    # define the new mode register
    remapped = np.linspace(0, n_spinorbitals - 1, n_spinorbitals, dtype=int)//2
    remapped[1::2] += int(np.ceil(n_spinorbitals / 2.))

    for term, coef in fermion_operator.terms.items():

        new_term = tuple([(int(remapped[ti[0]]), ti[1]) for ti in term])
        new_operator += FermionOperator(new_term, coef)

    return new_operator
