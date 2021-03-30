"""Utility functions to generate FermionOperators corresponding to the generalized UCCSD (UCCGSD)
ansatz for a designated number of orbitals or qubits. Excitations are grouped together to minimize
redundancy and symmetry-violating independence between related spin-orbitals."""

import numpy as np
from math import factorial
import itertools

from openfermion import FermionOperator

def hermitian_conjugate(terms):
    """Create nested list of tuples which can be utilized
    to generate FermionOperator associated with the Hermitian 
    conjugate of those same operators implied by input *terms*.

    For a product of fermionic terms, the Hermitian conjugate corresponds
    to a reverse ordering of the operators. The prefactor is the same, up to
    an overall sign, dictated by the number of permutations. 

    For singles and doubles, permutations are odd, so sign of prefactor is flipped.

    Args:
        terms (list): list of lists. Each sub-list is formatted as a tuple, and
            a tuple and a float. The tuples are themselves, nested tuple
            dictating spin-orbital index (int), and creation (1) or annihilation (0).
            e.g. one element in the terms list may be:
            [((0,1),(1,0),(3,1),(4,0)),0.5] = 0.5a_0^{\dagger}a_1a_3^{\dagger}a_4

    Returns:
        list: same format as input terms, with orbitals and operations inverted, and
            sign flipped.
    """
    try:
        return [[((v[0][3][0],1),
                (v[0][2][0],1),
                (v[0][1][0],0),
                (v[0][0][0],0)),
                -v[1]] for v in terms]
    except:
        raise ValueError('Input terms must be format as, e.g. [[((int,1),(int,0),(int,1),(int,0)),float],...]')


def get_spin_ordered(n_orbs, pp, qq, rr=-1, ss=-1, up_down=False):
    """For a given set of orbitals, provide corresponding spin-orbital indices,
    based on the desired spin-orbital ordering. Depending on user-convention, one
    typically orders spin-orbitals as all spin-up followed by all spin-down (this
    is the convention followed in e.g. qiskit), or alternating spin-up, spin-down
    (as done in e.g. openfermion). By setting *up_down* flag to True, one uses
    the qiskit convention, and False, the openfermion convention. 

    This function accommodates either a pair or quartet of orbital indices, and returns
    two tuples of int reflecting the spin-up fermion indices for desired orbitals, and
    the spin-down fermion indices for the same.

    Args:
        n_orbs (int): number of orbitals in the fermion basis (this is number of
            spin-orbitals divided by 2)
        pp (int): orbital index
        qq (int): orbital index
        rr (int): orbital index, only specified for four-fermion operator
        ss (int): orbital index, only specified for four-fermion operator
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions.

    Returns:
        up (tuple): tuple of int -- 4 if rr and ss kwargs specified, else 2
            corresponding to spin-up spin-orbitals
        down (tuple): tuple of int -- 4 if rr and ss kwargs specified, else 2
            corresponding to spin-down spin-orbitals
    """
    if type(n_orbs) != int:
        raise TypeError('Invalid datatype for number of orbitals.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals must be positive.')
    try:
        pp, qq, rr, ss = int(pp), int(qq), int(rr), int(ss) #Force orbital indices to int-type
    except TypeError:
        raise TypeError("All orbital indices (pp, qq, rr, ss) must be integer-type.")

    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')
    if up_down: #all spin up then all spin down
        up = pp, qq, rr, ss
        down = n_orbs + pp, n_orbs + qq, n_orbs + rr, n_orbs + ss
    else: #alternating spin up/spin down 
        up = 2*pp, 2*qq, 2*rr, 2*ss
        down = 2*pp +1, 2*qq +1, 2*rr +1, 2*ss +1
    if rr < 0: #if user has passed a 4-fermion operator
        return up[:2], down[:2]
    
    return up,down #if a user has passed a 2-fermion operator


def get_group_1_2(n_orbs, p, q, r, s, up_down=False):
    """Identify spin singlet and spin triplet -type 4-orbital 
    excitations. We group equivalent terms,along with their Hermitian
    conjugates to prepare a single FermionOperator which will obey
    spin-symmetries. We then generalize expressions of the form:

    a_p^{\dagger}a_q^{\dagger}a_r a_s with the constraints

    Either: (p < q and p < r and p < s and r < s) and (q != r and q != s)
    Or: (q == r and p < s) and (pp != qq and ss != qq). 
    
    Args:
        n_orbs (int): number of orbitals in basis (this is number of
            spin-orbitals divided by 2)
        p (int): orbital index
        q (int): orbital index
        r (int): orbital index
        s (int): orbital index
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions.        

    Returns:
        (list): list of input for FermionOperator corresponding to all
            singlet and triplet terms.
    """
    if type(n_orbs) != int:
        raise TypeError('Invalid datatype for number of orbitals.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals must be positive.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')

    up,dn = get_spin_ordered(n_orbs, p, q, r, s, up_down=up_down) #get spin-orbital indices

    #get triplet term
    triplet = [[((up[0], 1), (up[1], 1), (up[2],0), (up[3], 0)), 1.0],
                [((up[0], 1), (dn[1], 1), (up[2], 0), (dn[3], 0)), 0.5],
                [((up[0], 1), (dn[1], 1), (dn[2], 0), (up[3], 0)), 0.5],
                [((dn[0], 1), (up[1], 1), (up[2], 0), (dn[3], 0)), 0.5],
                [((dn[0], 1), (up[1], 1), (dn[2], 0), (up[3], 0)), 0.5],
                [((dn[0], 1), (dn[1], 1), (dn[2], 0), (dn[3], 0)), 1.0]]
    
    #get singlet term
    singlet = [[((up[0], 1), (dn[1], 1), (up[2], 0), (dn[3], 0)), 0.5],
                [((up[0], 1), (dn[1], 1), (dn[2], 0), (up[3], 0)), -0.5],
                [((dn[0], 1), (up[1], 1), (up[2], 0), (dn[3], 0)), -0.5],
                [((dn[0], 1), (up[1], 1), (dn[2], 0), (up[3], 0)), 0.5]]

    #add Hermitian conjugates 
    triplet += hermitian_conjugate(triplet)
    
    singlet += hermitian_conjugate(singlet)
    
    return singlet + triplet


def get_group_3_4(n_orbs, p, q, r, s, up_down=False):
    """Identify spin singlet -type 4-orbital 
    excitations. We group equivalent terms,along with their Hermitian
    conjugates to prepare a single FermionOperator which will obey
    spin-symmetries. We then generalize expressions of the form:

    a_p^{\dagger}a_q^{\dagger}a_r a_s with the constraints

    Either: (p == q and q != r and r != s and s != p) and r < s
    Or: p == q and q == r and r != s. 

    Note there is no spin-triplet term for these criteria.
    
    Args:
        n_orbs (int): number of orbitals in basis (this is number of
            spin-orbitals divided by 2)
        p (int): orbital index
        q (int): orbital index
        r (int): orbital index
        s (int): orbital index
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions.        

    Returns:
        singlet (list): list of input for FermionOperator corresponding to spin-singlet
            terms.
    """
    if type(n_orbs) != int:
        raise TypeError('Invalid datatype for number of orbitals.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals must be positive.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')

    up,dn = get_spin_ordered(n_orbs, p, q, r, s, up_down=up_down) #get spin-orbital indices

    #prepare combinations
    singlet = [[((up[0], 1), (dn[1], 1), (up[2], 0), (dn[3], 0)), 1.0],
                [((up[0], 1), (dn[1], 1), (dn[2], 0), (up[3], 0)), 1.0]]      
    
    #add Hermitian conjugate
    singlet += hermitian_conjugate(singlet)

    return singlet


def get_group_5(n_orbs, p, q, r, s, up_down=False):
    """Identify spin singlet -type 4-orbital 
    excitations. We group equivalent terms,along with their Hermitian
    conjugates to prepare a single FermionOperator which will obey
    spin-symmetries. We then generalize expressions of the form:

    a_p^{\dagger}a_q^{\dagger}a_r a_s with the constraints

    p == q and q != r and r == s and p < s

    Note there is no spin-triplet term for this criteria.
    
    Args:
        n_orbs (int): number of orbitals in basis (this is number of
            spin-orbitals divided by 2)
        p (int): orbital index
        q (int): orbital index
        r (int): orbital index
        s (int): orbital index
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions.        

    Returns:
        singlet (list): list of input for FermionOperator corresponding to spin-singlet
            terms.
    """
    if type(n_orbs) != int:
        raise TypeError('Invalid datatype for number of orbitals.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals must be positive.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')

    #get spin-orbital indices
    up,dn = get_spin_ordered(n_orbs, p, q, r, s, up_down=up_down)

    #prepare combinations
    singlet = [[((up[0], 1), (dn[1], 1), (up[2], 0), (dn[3], 0)), 2.0]]

    #add Hermitian conjugate terms
    singlet += hermitian_conjugate(singlet)

    return singlet


def get_doubles(n_orbs, up_down=False):
    """Prepare a list of all double-excitation terms in the UCCGSD for a given number of fermionic
    orbital basis states, grouped together according to spin-compensated
    spin-triplet or spin-singlet spin-orbital combinations. The number of 
    combinations of spin-orbitals for a given combination of orbital basis states,
    as well as the relative prefactors, depend on the specific orbital indices.
    For example, there are more ways to permute four distinct orbitals, than when
    two of the orbital indices are the same.

    Five different groupings of orbital indices can be identified, as in e.g. (arXiv:1911.10205)
    covering all possible combinations.

    Args:
        n_orbs (int): number of orbitals in basis (this is number of
            spin-orbitals divided by 2)
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions. 

    Returns:
        all_terms (list): list of lists. Each sub-list is formatted as a tuple, and
            a tuple and a float. The tuples are themselves, nested tuple
            dictating spin-orbital index (int), and creation (1) or annihilation (0).
            e.g. one element in the all_terms list may be:
            [((0,1),(1,0),(3,1),(4,0)),0.5] = 0.5a_0^{\dagger}a_1a_3^{\dagger}a_4
    """
    if type(n_orbs) != int:
        raise TypeError('Invalid datatype for number of orbitals.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals must be positive.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')

    all_terms = list()
    selection = np.linspace(0, n_orbs-1, n_orbs, dtype=int)
    for pp, qq, rr, ss in itertools.product(selection, repeat=4):

        if (pp < qq and pp < rr and pp < ss) and (rr < ss) and (qq != rr and qq != ss):
            terms = get_group_1_2(n_orbs, pp, qq, rr, ss, up_down=up_down)    
        elif qq == rr and pp < ss and pp != qq and ss != qq:
            terms = get_group_1_2(n_orbs, pp, qq, rr, ss, up_down=up_down)
        elif (pp == qq and qq != rr and rr != ss and ss != pp) and rr < ss:
            terms = get_group_3_4(n_orbs, pp, qq, rr, ss, up_down=up_down)
        elif pp == qq and qq == rr and rr != ss:
            terms = get_group_3_4(n_orbs, pp, qq, rr, ss, up_down=up_down)
        elif pp == qq and qq != rr and rr == ss and pp < ss:     
            terms = get_group_5(n_orbs, pp, qq, rr, ss, up_down=up_down)      
        else:
            continue #skip the append if no new terms defined by conditions above
        
        all_terms.append(terms)

    return all_terms


def get_singles(n_orbs, up_down=False):
    """Prepare a list of all single-excitation terms in the UCCGSD for a given number of fermionic
    orbital basis states, grouped together according to spin-compensated
    spin-orbital combinations. 

    Args:
        n_orbs (int): number of orbitals in basis (this is number of
            spin-orbitals divided by 2)
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions. 

    Returns:
        all_terms (list): list of lists. Each sub-list is formatted as a tuple, and
            a tuple and a float. The tuples are themselves, nested tuple
            dictating spin-orbital index (int), and creation (1) or annihilation (0).
            e.g. one element in the all_terms list may be:
            [((0,1),(1,0)),0.5] = 0.5a_0^{\dagger}a_1
    """
    if type(n_orbs) != int:
        raise TypeError('Invalid datatype for number of orbitals.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals must be positive.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')

    all_terms = list()
    selection = np.linspace(0, n_orbs-1, n_orbs, dtype = int) #all possible orbitals
    for pp, qq in itertools.product(selection, repeat=2): #iterate over all pairings of orbital-indices

        if qq <= pp: #avoid duplicates, take only lower-triangle
            continue

        up,down = get_spin_ordered(n_orbs, pp, qq, up_down=up_down) #get spin-orbital indices
        
        terms = [[((up[0], 1), (up[1],0)), 1.],
                 [((down[0], 1), (down[1],0)), 1.], #spin-compensated (up <> down)
                 [((up[1], 1), (up[0],0)), -1.], #Hermitian conjugate
                 [((down[1], 1), (down[0],0)), -1.]]  #h.c. of spin-compensated term         
        
        all_terms.append(terms)

    return all_terms


def get_all_excitations(n_orbs, up_down=False):
    """Enumerate all possible single and double excitations
    in the UCCGSD for a given number of fermionic orbital basis
    states. Prepare list of inputs to be applied as FermionicOperators.

    The output is a third-order nested list. The list represents groups of
    operators acting on related sets of orbitals. Each of these groups
    enumerate a single or double excitations, and the related coefficients
    between their constituent terms. This output is used to build a set of
    FermionOperator objects

    Args:
        n_orbs (int): number of orbitals in basis (this is number of
            spin-orbitals divided by 2)
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions. 

    Returns:
        all_terms (list): nested list as detailed above. Each element is a list
            over either a single or double excitation, represented by coefficients
            and the related operator indices and actions (creation/annihilation)
    """
    if type(n_orbs) != int:
        raise TypeError('Number of orbitals (n_orbs) must be integer type.')
    if n_orbs < 1:
        raise ValueError('Number of orbitals (n_orbs) must be at least 1.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')

    singles = get_singles(n_orbs, up_down=up_down) #get all single-excitations
    doubles = get_doubles(n_orbs, up_down=up_down) #get all double-excitations
    all_terms = singles + doubles #combine singles and doubles into one list
    
    return all_terms 


def uccgsd_generator(n_qubits, single_coeffs=None, double_coeffs=None, up_down=False):
    """Construct a list of FermionOperators enumerating the Unitary Coupled-Cluster
    Generalized Singles and Doubles excitations. Groups of spin-compensating and conjugate
    terms are combined together, with their relative weights to enforce spin-symmetries
    and reduce redundancy in VQE optimization parameters. All singles are followed by
    double excitations.

    Args:
        n_qubits (int): number of qubits (spin-orbitals) in basis (this is number of
            basis-orbitals multiplied by 2)
        up_down (bool): flag for using qiskit (True) or openfermion (False)
            spin-ordering conventions.

    Returns:
        all_operators (list): list of FermionOperator objects
    """
    if type(n_qubits) != int:
        raise TypeError('Number of qubits (n_qubits) must be integer type.')
    if n_qubits < 2:
        raise ValueError('Number of qubits (n_qubits) must be at least 2.')
    elif np.mod(n_qubits,2) != 0:
        raise ValueError('Invalid number of qubits (n_qubits) -- must be even.')
    if type(up_down) != bool:
        raise TypeError('Spin-ordering arg (up_down) must be boolean.')
    
    coeffs = get_coeffs(n_qubits, single_coeffs, double_coeffs) #check coeffecients passed, or generate random ones
    operators = get_all_excitations(n_qubits//2, up_down=False) #get all operator input arguments

    all_operators = list()
    for index,oi in enumerate(operators):
        current= FermionOperator() #create new FermionOperator
        for term in oi:
            current += FermionOperator(*term[:-1], term[-1]*coeffs[index]) #add term to operator

        all_operators.append(current) #add to list of total 

    return all_operators


def get_singles_number(n_orbitals):
    """Get number of independent terms in the set of singles excitations
    for a designated number of orbital states. Note argument is the number of 
    spatial orbital basis states, number of spin orbitals is twice as large.

    Args:
        n_orbitals(int): number of basis orbitals

    Return:
        int number of singles excitations
    """
    if n_orbitals != np.floor(n_orbitals):
        raise ValueError('Number of orbitals must be integer valued')

    try:
        return n_orbitals*(n_orbitals - 1)//2
    except:
        raise ValueError('Invalid format for number of orbitals, expecting integer.')


def get_doubles_number(n_orbitals):
    """Get number of independent terms in the set of doubles excitations
    for a designated number of orbital states. Note argument is the number of 
    spatial orbital basis states, number of spin orbitals is twice as large.

    Args:
        n_orbitals(int): number of basis orbitals

    Return:
        int number of doubles excitations
    """
    if np.mod(n_orbitals, 1) != 0.0:
        raise ValueError('Number of orbitals must be integer valued')

    try:
        return n_orbitals * ( n_orbitals**3 + 2*n_orbitals**2 - n_orbitals - 2 ) // 8
    except:
        raise TypeError('Invalid format for number of orbitals, expecting integer.')


def get_excitation_number(n_orbitals):
    """Get number of independent singles and doubles excitations for a given
    number of basis orbital (spatial) states. Note input argument is half the 
    number of spin-orbitals.

    Args:
        n_orbitals(int): number of basis orbitals

    Return:
        int number of excitations
    """

    return get_singles_number(n_orbitals) + get_doubles_number(n_orbitals)


def get_coeffs(n_qubits, single_coeffs=None, double_coeffs=None):
    """Prepare coefficients for UCCGSD excitation terms. User has option
    to either pass specified singles and doubles excitation coefficients,
    or to utilize randomly chosen values. Note, if passing own values, check
    that number is consistent with the output for *get_singles_number* and
    *get_doubles_number* before proceeding.
    
    Args:
        n_qubits (int): number of qubits for problem (2x number of orbitals)
        single_coeffs (list or numpy array, OR None): initial singles-excitation
            coefficients, or skip.
        double_coeffs (list or numpy array, OR None)): initial doubles-excitation
            coefficients, or skip.
    Returns:
        coeffs (numpy array of float): coefficients for initializing the FermionOperator
            for the UCCGSD.
    """
    if single_coeffs is None:
        single_coeffs = np.random.random(get_singles_number(n_qubits//2))
    elif len(single_coeffs) != get_singles_number(n_qubits//2):
        raise ValueError('Invalid number of single excitation coefficients.')

    if double_coeffs is None:
        double_coeffs = np.random.random(get_doubles_number(n_qubits//2))
    elif len(double_coeffs) != get_doubles_number(n_qubits//2):
        raise ValueError('Invalid number of double excitation coefficients.')

    coeffs = np.concatenate((single_coeffs, double_coeffs))

    return coeffs
