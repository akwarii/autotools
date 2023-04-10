import numpy as np
from matplotlib import pyplot as plt

from autotools.utils import get_atoms
from autotools.calculations import MinimizationCalculator

from deepmd.calculator import DP

from ase import Atoms
from ase.optimize import BFGS
from ase.eos import EquationOfState
from ase.io.trajectory import Trajectory

plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'serif'


class EOSCalculator:
    def __init__(self, atoms_list: list[Atoms]) -> None:
        """Interface to ASE for equation-of-state calculations.

        atoms_list: list of ASE Atoms
            Systems to calculate EOS for.
        calculator: str
            Name of the ASE calculator to use.
        setup: dict
           Dictionary of the parameters to use for the calculator.
        trajectory: Trjectory object or str
            Write configurations to a trajectory file.
        callback: function
            Called after every energy calculation.
        """
        self.atoms_list = atoms_list




def main():
    filename = ['examples/STRUCT/data.TETRA']
    calc = DP(model='examples/POTENTIALS/ZrO2_11_1-compressed.pb')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap('tab10')
    for i, file in enumerate(filename):
        # Prepare atoms object
        atoms = get_atoms(file, calc)
        
        # Minimize the initial structure
        minimizer = MinimizationCalculator(atoms)
        minimizer.run(symprec=1e-5)
        
        eos = calculate_eos(atoms, npoints=21, eps=0.08)
        
        plotdata = eos.getplotdata()
        label = str(file).split('.')[-1]
        ax = plot(*plotdata[-4:], len(atoms), label, cmap(i))
        fig = ax.get_figure()
        fig.savefig('eos.png')
        eos.plot()


def plot(x, y, v, e, nat, label, color, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    ax.plot(x / nat, y / nat, ls='-', color=color, label=label, linewidth=3)
    ax.plot(v / nat, e / nat, ls='', marker='o', mec=color, mfc=color, fillstyle='none', markersize=10)

    ax.set_xlabel(u'volume [Å$^3$/atom]')
    ax.set_ylabel(u'energy [eV/atom]')
    ax.legend(fontsize=20, ncols=2)

    return ax

def calculate_eos(atoms: Atoms, npoints: int = 10, eps: float = 0.04, fmax: float = 1e-10, steps: int = 1000, trajectory: str = None, logfile: str = None) -> EquationOfState:
    # Save original positions and cell
    p0 = atoms.get_positions()
    c0 = atoms.get_cell()

    if isinstance(trajectory, str):
        trajectory = Trajectory(trajectory, 'w', atoms)

    if trajectory is not None:
        trajectory.set_description({'type': 'eos',
                                    'npoints': npoints,
                                    'eps': eps})

    try:
        energies = []
        volumes = []
        for x in np.linspace(1 - eps, 1 + eps, npoints)**(1 / 3):
            atoms.set_cell(x * c0, scale_atoms=True)
            
            with BFGS(atoms, logfile=logfile) as opt:
                opt.run(fmax=fmax, steps=steps)
            
            volumes.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())
            
            if trajectory is not None:
                trajectory.write()
            
        return EquationOfState(volumes, energies, eos='birchmurnaghan')
    finally:
        atoms.cell = c0
        atoms.positions = p0
        if trajectory is not None:
            trajectory.close()

if __name__ == "__main__":
    main()