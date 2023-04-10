from pathlib import Path

from deepmd.calculator import DP

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.calculator import Calculator


AVAILABLE_CALC = {'dp': DP, 'lammps': LAMMPS, 'vasp': Vasp}

def get_calculator(self, calc: str, setup: dict) -> Calculator:
    calc = calc.lower()

    if calc not in AVAILABLE_CALC.keys():
        raise ValueError(f"Calculator {self.calc} not supported.")

    return AVAILABLE_CALC[calc](**setup)


def get_path_from_regex(patterns: list[str]) -> list[Path]:
    files = []
    for p in patterns:
        p = Path(p)
        for f in p.parent.glob(p.name):
            files.append(f.absolute())
    return sorted(files)


def mass_to_symbol(x):
    from ase.data import chemical_symbols, atomic_masses
    return chemical_symbols[list(atomic_masses).index(x)]


def get_atoms(filename: str) -> Atoms:
    from ase.io import read
    from ase.io.vasp import read_vasp
    from ase.io.lammpsdata import read_lammps_data
    from ase.io.formats import UnknownFileTypeError

    try:
        atoms = read_lammps_data(filename, style='atomic')

        # Set chemical symbols
        symbols = [mass_to_symbol(m) for m in atoms.get_masses().round(3)]
        atoms.set_chemical_symbols(symbols)
    except UnknownFileTypeError:
        try:
            atoms = read_vasp(filename)
        except UnknownFileTypeError:
            atoms = read(filename)

    return atoms
