from typing import Any
import numpy as np

from ase import Atoms
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry


class MinimizationCalculator:
    def __init__(self, atoms: Atoms, logfile: str = None, trajectory: str = None) -> None:
        """Minimalistic interface to ASE optimizer.

        atoms: ASE Atoms
            System to minimize
        logfile: file objct or str
           If *logfile* is a string, a file with that name will be opened. Use '-' for stdout and None to disable.
        trajectory: Trjectory object or str
            If *trajectory* is a string, a Trajectory file with that name will be opened. Use None to disable.
        """
        self.atoms = atoms
        self.logfile = logfile
        self.trajectory = trajectory
        
    def run(self, relax_cell: bool = True, fmax: float = 1e-10, steps: int = 1000, symprec: float = 0) -> None:
        """Run minimization using BFGS. A wargning will be raised if the minimization does not converge.
        If symprec is set, the symmetry will be fixed during the minimization.
        
        relax_cell: bool
            Whether to relax the cell or not.
        fmax: float
            Force convergence criterion.
        steps: int
            Maximum number of steps.
        symprec: float
            Symmetry precision. Set to 0 to disable symmetry.
        """
        
        if isinstance(self.trajectory, str):
            self.trajectory = Trajectory(self.trajectory, 'w', self.atoms)

        if self.trajectory is not None:
            self.trajectory.set_description({'type': 'relax',
                                             'fmax': fmax,
                                             'steps': steps,
                                             'relax_cell': relax_cell})
        
        # Minimize forces and stress
        if relax_cell:
            ecf = ExpCellFilter(self.atoms,
                                hydrostatic_strain=False,
                                constant_volume=False,
                                scalar_pressure=0.0)
        else:
            ecf = self.atoms

        # Fix symmetry
        if symprec == 0:
            fixsym = FixSymmetry(self.atoms, symprec=symprec, verbose=False)
            self.atoms.set_constraint(fixsym)

        # Run minimization using BFGS
        with BFGS(ecf, logfile=self.logfile) as opt:
            if self.trajectory is not None:
                opt.attach(self.trajectory)
            opt.run(fmax=fmax, steps=steps)

            if self.trajectory is not None:
                self.trajectory.close()

        self.atoms.constraints = None

        return self.atoms
    
    def write_atoms(self, filename: str, format: str, **kwargs: Any) -> None:
        """Write the current configuration to a file.
        
        filename: str
            Name of the file to write to.
        style: str
            Force convergence criterion.
        steps: int
            Maximum number of steps.
        symprec: float
            Symmetry precision. Set to None to disable symmetry.
        """
        style = style.lower()
        if style in ['lammps', 'lmp', 'lammps-data']:
            from ase.io.lammpsdata import write_lammps_data
            
            # Write final structure to file and add masses to LAMMPS data file
            write_lammps_data(filename, self.atoms, **kwargs)

            lmp_data = open(filename, "r").readlines()
            for i, line in enumerate(lmp_data):
                if line.find("Atoms") != -1:
                    line_number = i
                    break

            # Add masses to LAMMPS data file
            lmp_data.insert(line_number - 1, f"Masses\n\n")
            for i, mass in enumerate(np.unique(self.atoms.get_masses())):
                lmp_data.insert(line_number + i, f"{i + 1} {mass:.3f}\n")

            with open(filename, "w") as f:
                f.writelines(lmp_data)
            write_lammps_data(filename, self.atoms, **kwargs)
        else:
            from ase.io import write
            write(filename, self.atoms, format=format, **kwargs)