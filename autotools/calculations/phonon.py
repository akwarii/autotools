from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import ArrayLike

from deepmd.calculator import DP

from ase import Atoms
from ase.optimize import FIRE
from ase.io.vasp import write_vasp
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.trajectory import Trajectory
from ase.constraints import ExpCellFilter
from ase.spacegroup.symmetrize import FixSymmetry
from ase.data import chemical_symbols, atomic_masses

from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints
from phonopy.interface.calculator import read_crystal_structure


class PhononCalculator:
    def __init__(self) -> None:
        pass


def main() -> None:
    datafile = "data.TETRA"
    potfile = "examples/POTENTIALS/ZrO2_11_1-compressed.pb"
    calc = DP(model=potfile)

    mesh = 100
    supercell = np.eye(3) * 3

    atoms = get_atoms(datafile, calc)
    
    # Get band path
    bandpath = atoms.cell.bandpath()
    klabels = bandpath.path.split(',')[0]
    kpts = bandpath.special_points
    band_path = [[[k for k in kpts[l]] for l in klabels]]
    labels = ' '.join(klabels).replace('G', r'$\Gamma$').split()

    # Minimize the initial structure
    atoms = run_minimize(atoms, datafile, fmax=1e-10, steps=1000, traj=False)

    # Create Phonopy object
    write_vasp("POSCAR-unitcell", atoms, direct=True, sort=True)
    cell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode='vasp')
    phonon = Phonopy(cell, supercell, primitive_matrix="auto")
    
    # Generate force constants
    phonon.generate_displacements()
    set_of_forces = get_disp_forces(phonon, calc)
    phonon.produce_force_constants(forces=set_of_forces)
    phonon.save()

    # Get phonon bands and DOS and save them
    phonon.run_mesh(mesh)
    run_bands(phonon, band_path, labels=labels)
    run_dos(phonon)

    Path("POSCAR-unitcell").unlink()
    subprocess.Popen("phonoto-bandplot band.yaml --dos total_dos.dat -o band.png", shell=True)


def get_atoms(datafile: str, calc) -> Atoms:
    atoms = read_lammps_data(datafile, style='atomic')

    mass_to_symbol = lambda x: chemical_symbols[list(atomic_masses).index(x)]
    symbols = [mass_to_symbol(m) for m in atoms.get_masses().round(3)]
    atoms.set_chemical_symbols(symbols)

    atoms.calc = calc
    return atoms


def run_minimize(atoms: Atoms, filename: str, fmax: float = 1e-10, steps: int = 1000, traj: bool = False) -> None:
    # Minimize forces and stress
    ecf = ExpCellFilter(atoms,
                        hydrostatic_strain=False,
                        constant_volume=False,
                        scalar_pressure=0.0)

    # Fix symmetry
    fixsym = FixSymmetry(atoms, symprec=1e-10, verbose=True)
    atoms.set_constraint(fixsym)

    # Run minimization using FIRE
    fire = FIRE(ecf)
    traj = Trajectory("relax.traj", "w", atoms)
    fire.attach(traj)
    fire.run(fmax=fmax, steps=steps)

    # Update atoms object
    traj.close()
    traj = Trajectory("relax.traj")
    atoms = traj[-1]
    traj.close()

    atoms.constraints = None

    if not traj:
        Path("relax.traj").unlink()

    # Write final structure to file and add masses to LAMMPS data file
    write_lammps_data(filename, atoms, atom_style='atomic')

    lmp_data = open(filename, "r").readlines()
    for i, line in enumerate(lmp_data):
        if line.find("Atoms") != -1:
            line_number = i
            break

    # Add masses to LAMMPS data file
    lmp_data.insert(line_number - 1, f"Masses\n\n")
    for i, mass in enumerate(np.unique(atoms.get_masses())):
        lmp_data.insert(line_number + i, f"{i + 1} {mass:.3f}\n")

    with open(filename, "w") as f:
        f.writelines(lmp_data)

    return atoms


def get_disp_forces(phonon: Phonopy, calc) -> list[np.ndarray]:
    supercells = phonon.get_supercells_with_displacements()

    # Force calculations by calculator
    set_of_forces = []
    for scell in supercells:
        cell = Atoms(symbols=scell.get_chemical_symbols(),
                     scaled_positions=scell.get_scaled_positions(),
                     cell=scell.get_cell(),
                     calculator=calc,
                     pbc=True)
        
        forces = cell.get_forces()
        drift_force = forces.sum(axis=0)
        print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))

        # Simple translational invariance
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    return set_of_forces


def run_bands(phonon: Phonopy, band_path: ArrayLike, labels: list[str] = None, npoints: int = 101) -> None:
    bands = get_band_qpoints(band_path, npoints=npoints)
    phonon.run_band_structure(bands, labels=labels)

    band_structure = phonon.band_structure
    band_structure.write_yaml()


def run_dos(phonon: Phonopy) -> None:
    phonon.run_total_dos()
    dos = phonon.total_dos
    dos.write()


if __name__ == "__main__":
    main()
