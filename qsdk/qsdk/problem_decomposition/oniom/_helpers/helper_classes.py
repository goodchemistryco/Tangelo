"""Fragment class, used in construction of ONIOM layers -- contains details of both the
constituent geometry (i.e. which atoms from system are in fragment, which bonds are broken
and how to fix them) as well as the solver(s) to use.
"""

import numpy as np
from pyscf import gto

# Imports of electronic solvers.
from qsdk.toolboxes.molecular_computation.integral_calculation import prepare_mf_RHF
from qsdk.electronic_structure_solvers import CCSDSolver, FCISolver, VQESolver, MINDO3Solver


class Fragment:

    def __init__(self, solver_low, options_low=None, solver_high=None, options_high=None, selected_atoms=None, charge=0, spin=0, broken_links=None):
        """Fragment class for the ONIOM solver. Each fragment can have broken links.
        In this case, they are capped with a chosen atom. Each fragment can also have
        up to two solvers (low and high accuracy).

        Args:
            solver_low (str): Specification of low accuracy solver for fragment.
            options_low (str): Specification of low accuracy solver options.
            solver_high (str): Specification of higher accuracy solver for fragment.
            options_high (str): Specification of higher accuracy solver options.
            selected_atoms (list of int or int): Which atoms from molecule are in fragment. int counts from start of xyz.
            spin (int): Spin associated witht this fragment.
            charge (int): Charge associated witht this fragment.
            broken_links (list of Link): Bonds broken when forming fragment.

        Attributes:
            mol_low (pyscf.gto.Mole): PySCF molecule of this fragment, to be solved with low accuracy.
            mol_high (pyscf.gto.Mole): PySCF molecule of this fragment, to be solved with high accuracy.
            e_fragment (float): Energy of this fragment as defined by ONIOM. None if simulate
                has not yet been called.
        """

        default_solver_options = {"basis": "sto-3g"}

        # Check to see if a fragment has no solver_high when only a portion of a molecule
        # is selected. If this is allowed, the energy of a fragment is added to the
        # system (all system with solver_low + fragment with solver_low), adding
        # more atoms in the system than there are.
        if selected_atoms is not None and solver_high is None:
            raise RuntimeError("If atoms are selected (selected_atoms different than None), a solver_high must be provided.")

        self.selected_atoms = selected_atoms

        # Solver with low accuracy.
        self.solver_low = solver_low.upper()
        self.options_low = options_low if options_low is not None else default_solver_options

        # Solver with higher accuracy.
        self.solver_high = solver_high.upper() if solver_high is not None else solver_high
        self.options_high = options_high if options_high is not None else default_solver_options

        # For this fragment (not the whole molecule).
        self.geometry = None
        self.spin = spin
        self.charge = charge

        self.broken_links = broken_links
        self.mol_low = None
        self.mol_high = None
        self.e_fragment = None
        self.solvers = list()

    def simulate(self):
        """Get the solver object for this layer.

        Returns:
            float: Energy for the fragment.
        """

        # Low accuracy solver.
        # We begin by defining the molecule.
        if self.mol_low is None:
            self.mol_low = self.get_mol(self.options_low["basis"])
            # Basis is only relevant when making the pyscf molecule. After this,
            # it is discarded (not needed for electronic solver because they retrieved
            # it from the molecule object).
            self.options_low = {i:self.options_low[i] for i in self.options_low if i!="basis"}

        e_low = self.get_energy(self.mol_low, self.solver_low, self.options_low)

        # Higher accuracy solver.
        e_high = 0.
        if self.solver_high is not None:

            # Molecule is reconstructed (in case a different basis is used).
            if self.mol_high is None:
                self.mol_high = self.get_mol(self.options_high["basis"])
                # Same process done as in low accuracy process.
                self.options_high = {i:self.options_high[i] for i in self.options_high if i!="basis"}

            e_high = self.get_energy(self.mol_high, self.solver_high, self.options_high)

            # Contribution from low accuracy is substracted, as defined by ONIOM.
            e_low *= -1

        self.e_fragment = e_high + e_low
        return self.e_fragment

    def get_mol(self, basis):
        """Get the molecule object for this fragment (with a specified basis).

        Returns:
            pyscf.gto.Mole: Molecule object.
        """

        mol= gto.Mole()
        mol.build(atom=self.geometry,
                  basis=basis,
                  charge=self.charge,
                  spin=self.spin)

        return mol

    def update_geometry(self, new_geometry):
        """Update the geometry of the fragment. If there is already a mol
        object built, update it with the appropriate method.

        Args:
            new_geometry (list): XYZ atomic coords in [[str, (float, float,
                float)], ...].
        """

        self.geometry = new_geometry

        if self.mol_low is not None:
            self.mol_low.set_geom_(new_geometry, inplace=True)

        if self.mol_high is not None:
            self.mol_high.set_geom_(new_geometry, inplace=True)

    def get_energy(self, molecule, solver, options_solver):
        """Get the solver object for this layer.

        Args:
            molecule (pyscf.gto): Molecule for this fragment (with repaired links).
            sovler (str): Which solver to use.
            options_solver (dict): Options for the solver.

        Returns:
            float: Energy of the fragment.
        """

        if solver == "RHF":
            solver = prepare_mf_RHF(molecule, **options_solver)
            energy = solver.e_tot
        elif solver == "CCSD":
            solver = CCSDSolver()
            energy = solver.simulate(molecule, **options_solver)
        elif solver == "FCI":
            solver = FCISolver()
            energy = solver.simulate(molecule, **options_solver)
        elif solver == "MINDO3":
            solver = MINDO3Solver()
            energy = solver.simulate(molecule, **options_solver)
        elif solver == "VQE":
            molecule_options = {'molecule': molecule}
            solver = VQESolver({**molecule_options, **options_solver})
            solver.build()
            energy = solver.simulate()
        else:
            raise NotImplementedError("This {} solver has not been implemented yet in ONIOMProblemDecomposition".format(solver))

        # Storing the solver object after the energy computation.
        self.solvers.append(solver)

        return energy

    def get_scanners(self):
        """Get the scanner necessary for either gradient, or energy-only associated
        with this layer's solver.
        """

        self.grad_scanners = list()
        for solver in self.solvers:
            self.grad_scanners.append(solver.nuc_grad_method().as_scanner())


class Link:

    def __init__(self, staying, leaving, factor=1.0, species='H'):
        """Bonds broken during the layer-construction process in ONIOM must be mended.
        This class represents a broken-bond link, and has associated methods to generate
        a new bond, appending the intended species.

        Args:
            index1 (int): Order in the molecular geometry of atom retained in model-unit.
            leaving (int): Order in mol. Geometry of atom lost.
            factor (float) optional: Rescale length of bond, from that in the original molecule.
            species (str) optional: Atomic species of appended atom for new link.
        """

        self.staying = staying
        self.leaving = leaving
        self.factor = factor
        self.species = species

    def relink(self, geometry):
        """Create atom at location of mended-bond link.

        Args:
            geometry (list of positions): Atomic positions in format [[str,tuple(float,float,float)],...].

        Returns:
            Atomic species, and position of replacement atom.
        """

        staying = np.array(geometry[self.staying][1])
        leaving = np.array(geometry[self.leaving][1])
        replacement = self.factor*(leaving-staying) + staying

        return (self.species, (replacement[0], replacement[1], replacement[2]))
