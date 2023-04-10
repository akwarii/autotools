import os
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from re import findall, DOTALL
from contextlib import contextmanager

import numpy as np

import pymatgen
import clusterlib
import custodian

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger("AutoTools")

LOGLEVEL = logging.DEBUG
AVAIL_CPU = len(os.sched_getaffinity(0))


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color
        self.color_map = {
            'DEBUG': '\033[1;36m',  # cyan
            'INFO': '\033[1;32m',  # green
            'WARNING': '\033[1;33m',  # yellow
            'ERROR': '\033[1;31m',  # red
            'CRITICAL': '\033[1;31m',  # bright red
            'RESET': '\033[0m'  # Reset to default color
        }

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in self.color_map:
            levelcolor = self.color_map.get(
                levelname, self.color_map[record.levelname])
            resetcolor = self.color_map['RESET']
            record.levelname = f'{levelcolor}[{levelname}]{resetcolor}'
        return super().format(record)


class LogFile:
    def __init__(self, logfile):
        self.keywords = []
        self.run_logs = []
        self.logfile = logfile
        self.elastic_tensor = np.zeros((6, 6))
        
        self._set_run_logs()
        self._get_elastic_tensor()
        
        self.runs = len(self.run_logs)

    def _set_run_logs(self):
        pattern = 'Mbytes.*?\n(.*?)Loop'
        logfile_str = open(self.logfile, 'r').read()
        
        for match in findall(pattern, logfile_str, flags=DOTALL):
            lines = match.splitlines()
            
            current_keywords = lines[0].split()
            current_values = np.array([line.split() for line in lines[1:]]).astype(float)
            
            self.keywords.append(current_keywords)
            self.run_logs.append(dict(zip(current_keywords, current_values.T)))

    def get_thermo(self, thermo, run=-1):
        run_log = self.run_logs[run]
        return run_log[thermo] if np.abs(run) <= self.runs and thermo in run_log.keys() else None

    def get_perf(self, run=None):
        ...

    def _get_elastic_tensor(self):
        pattern = r'(?<=C\d\dall = )-?[\d\.]+(?:[eE][-+]?\d+)?'
        logfile_str = open(self.logfile, 'r').read()
        
        vect = np.array([match for match in findall(pattern, logfile_str, flags=DOTALL)]).astype(float)
        
        if vect is None:
            return
        
        vect[np.isclose(vect, 0, atol=1e-2)] = 0
        
        upper_indices = np.triu_indices(6)
        
        self.elastic_tensor[upper_indices] = vect
        self.elastic_tensor.T[upper_indices] = vect
        
    def avail_thermo(self, run=-1):
        return self.keywords[run] if np.abs(run) <= self.runs else None
    

def configure_logger(level: int, use_file=True, use_stream=True) -> None:
    logger.setLevel(level)

    if use_stream:
        stream_formatter = ColoredFormatter(
            '%(name)s :: %(asctime)s :: %(levelname)s :: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        sh = logging.StreamHandler()
        sh.setLevel(LOGLEVEL)
        sh.setFormatter(stream_formatter)
        logger.addHandler(sh)
    
    if use_file:
        file_formatter = logging.Formatter(
            '%(name)s :: %(asctime)s :: %(levelname)s :: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler('autotools.log')
        fh.setLevel(LOGLEVEL)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A program used to automatically create input files for LAMMPS and run calculations."
    )
    parser.add_argument(
        'conf_file',
        help="The YAML configuration file used to generate input files.",
    )

    return parser.parse_args(['config.yaml'])


def load_config(conf_file: str | Path) -> dict:
    logger.info('Configuration file is loaded.')
    with open(conf_file, 'r') as f:
        return load(f, Loader=Loader)


@contextmanager
def change_dir(path: str | Path):
    """Change the current working directory to the given path and return to the original working directory when done."""
    cwd = Path.cwd()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path = path.resolve()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def _set_env_with_default(key: str, value: str, default: str):
    if os.environ.get(key) is None:
        os.environ[key] = default
        logger.info(f"Environment variable {key} is empty. Use the default value {default}")
    else:
        os.environ[key] = str(config[key])
        logger.debug(f"Environment variable {key} is set to {os.environ[key]}")
        

def set_env(envconfig):
    _set_env_with_default('OMP_NUM_THREADS',
                         envconfig['OMP_NUM_THREADS'], '1')
    
    _set_env_with_default('TF_INTER_OP_PARALLELISM_THREADS',
                         envconfig['TF_INTER_OP_PARALLELISM_THREADS'], '0')
    
    _set_env_with_default('TF_INTRA_OP_PARALLELISM_THREADS',
                         envconfig['TF_INTRA_OP_PARALLELISM_THREADS'], '0')


def get_path_from_regex(pattern: list[str]) -> list[Path]:
    files = []
    for p in pattern:
        p = Path(p)
        for f in p.parent.glob(p.name):
            files.append(f.absolute())
    return sorted(files)


def get_pot_and_struct(pot_path: Path, struct_path: Path) -> None:
    pot_file = pot_path.name
    struct_file = struct_path.name
    if not Path(pot_file).exists():
        logger.debug(f'Create symlink {pot_file} -> {pot_path}')
        Path(pot_file).symlink_to(pot_path)
        
    if not Path(struct_file).exists():
        logger.debug(f'Copy file {struct_path} as {struct_file}')
        shutil.copyfile(struct_path, struct_file)


def make_lattice_input(file: Path, potential: Path, relax_box: bool, min_style: str, press: float, steps: int, etol: float, ftol: float) -> None:
    infile = 'units metal\n'
    infile += 'dimension 3\n'
    infile += 'atom_style atomic\n'
    infile += 'atom_modify map array\n'
    infile += 'neighbor 0.3 bin\n'
    infile += 'neigh_modify once no every 1 delay 0 check yes\n'
    infile += 'boundary p p p\n'
    infile += '\n'
    infile += f'read_data {file}\n'
    infile += '\n'
    infile += f'pair_style deepmd {potential}\n'
    infile += 'pair_coeff * *\n'
    infile += '\n'
    infile += 'thermo_style custom step fnorm pe pxx pyy pzz pxy pyz pxz\n'
    infile += 'thermo 1\n'
    infile += '\n'

    if relax_box:
        infile += 'min_style cg\n'
    else:
        infile += f'min_style {min_style}\n'
    infile += 'min_modify norm inf\n'

    if relax_box:
        infile += f'fix f99 all box/relax tri {press}\n'

    infile += '\n'
    infile += f'minimize {etol} {ftol} {steps} {2*steps}\n'

    if relax_box:
        infile += 'unfix f99\n'

    infile += '\n'
    infile += 'thermo_style custom pe cella cellb cellc cellalpha cellbeta cellgamma\n'
    infile += 'run 0\n'
    infile += '\n'
    infile += 'change_box all x final 0 $(lx) y final 0 $(ly) z final 0 $(lz) remap\n'
    infile += f'write_data {file}\n'

    p = Path('in.lattice')
    logger.info(f"Write {p} in {os.getcwd()}")
    p.write_text(infile)


def make_elastic_input(file: Path, potential: Path, strain: float, steps: int, etol: float, ftol: float) -> None:
    def init_mod_str() -> str:
        initfile = 'units metal\n'
        initfile += 'boundary p p p\n'
        initfile += '\n'
        initfile += f'read_data {file}\n'
        initfile += '\n'
        initfile += f'variable strain equal {strain}\n'
        initfile += f'variable etol equal {etol}\n'
        initfile += f'variable ftol equal {ftol}\n'
        initfile += f'variable steps equal {steps}\n'
        initfile += f'variable fsteps equal {2*steps}\n'
        initfile += '\n'
        initfile += 'variable cfac equal 1.0e-4\n'
        initfile += 'variable cunits string GPa\n'
        return initfile

    def potential_mod_str() -> str:
        potfile = 'neighbor 0.3 bin\n'
        potfile += 'neigh_modify once no every 1 delay 0 check yes\n'
        potfile += '\n'
        potfile += f'pair_style deepmd {potential}\n'
        potfile += 'pair_coeff * *\n'
        potfile += '\n'
        potfile += 'min_style cg\n'
        potfile += 'min_modify norm inf\n'
        potfile += 'min_modify dmax 1.0e-2 line quadratic\n'
        potfile += '\n'
        potfile += 'thermo 1\n'
        potfile += 'thermo_style custom step fmax pe press pxx pyy pzz pxy pxz pyz lx ly lz vol\n'
        potfile += 'thermo_modify norm no\n'
        return potfile

    def displace_mod_str() -> str:
        dispfile = 'if "${dir} == 1" then "variable len0 equal ${lx0}"\n'
        dispfile += 'if "${dir} == 2" then "variable len0 equal ${ly0}"\n'
        dispfile += 'if "${dir} == 3" then "variable len0 equal ${lz0}"\n'
        dispfile += 'if "${dir} == 4" then "variable len0 equal ${lz0}"\n'
        dispfile += 'if "${dir} == 5" then "variable len0 equal ${lz0}"\n'
        dispfile += 'if "${dir} == 6" then "variable len0 equal ${ly0}"\n'
        dispfile += '\n'
        dispfile += 'clear\n'
        dispfile += 'box tilt large\n'
        dispfile += 'read_restart restart.equil\n'
        dispfile += 'include mod.potential\n'
        dispfile += '\n'
        dispfile += 'variable delta equal -${strain}*${len0}\n'
        dispfile += 'variable deltaxy equal -${strain}*xy\n'
        dispfile += 'variable deltaxz equal -${strain}*xz\n'
        dispfile += 'variable deltayz equal -${strain}*yz\n'
        dispfile += 'if "${dir} == 1" then "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\n'
        dispfile += 'if "${dir} == 2" then "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\n'
        dispfile += 'if "${dir} == 3" then "change_box all z delta 0 ${delta} remap units box"\n'
        dispfile += 'if "${dir} == 4" then "change_box all yz delta ${delta} remap units box"\n'
        dispfile += 'if "${dir} == 5" then "change_box all xz delta ${delta} remap units box"\n'
        dispfile += 'if "${dir} == 6" then "change_box all xy delta ${delta} remap units box"\n'
        dispfile += '\n'
        dispfile += 'minimize ${etol} ${ftol} ${steps} ${fsteps}\n'
        dispfile += '\n'
        dispfile += 'variable tmp equal pxx\n'
        dispfile += 'variable pxx1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pyy\n'
        dispfile += 'variable pyy1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pzz\n'
        dispfile += 'variable pzz1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pxy\n'
        dispfile += 'variable pxy1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pxz\n'
        dispfile += 'variable pxz1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pyz\n'
        dispfile += 'variable pyz1 equal ${tmp}\n'
        dispfile += '\n'
        dispfile += 'variable C1neg equal ${d1}\n'
        dispfile += 'variable C2neg equal ${d2}\n'
        dispfile += 'variable C3neg equal ${d3}\n'
        dispfile += 'variable C4neg equal ${d4}\n'
        dispfile += 'variable C5neg equal ${d5}\n'
        dispfile += 'variable C6neg equal ${d6}\n'
        dispfile += '\n'
        dispfile += 'clear\n'
        dispfile += 'box tilt large\n'
        dispfile += 'read_restart restart.equil\n'
        dispfile += 'include mod.potential\n'
        dispfile += '\n'
        dispfile += 'variable delta equal ${strain}*${len0}\n'
        dispfile += 'variable deltaxy equal ${strain}*xy\n'
        dispfile += 'variable deltaxz equal ${strain}*xz\n'
        dispfile += 'variable deltayz equal ${strain}*yz\n'
        dispfile += 'if "${dir} == 1" then "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\n'
        dispfile += 'if "${dir} == 2" then "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\n'
        dispfile += 'if "${dir} == 3" then "change_box all z delta 0 ${delta} remap units box"\n'
        dispfile += 'if "${dir} == 4" then "change_box all yz delta ${delta} remap units box"\n'
        dispfile += 'if "${dir} == 5" then "change_box all xz delta ${delta} remap units box"\n'
        dispfile += 'if "${dir} == 6" then "change_box all xy delta ${delta} remap units box"\n'
        dispfile += '\n'
        dispfile += 'minimize ${etol} ${ftol} ${steps} ${fsteps}\n'
        dispfile += '\n'
        dispfile += 'variable tmp equal pe\n'
        dispfile += 'variable e1 equal ${tmp}\n'
        dispfile += 'variable tmp equal press\n'
        dispfile += 'variable p1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pxx\n'
        dispfile += 'variable pxx1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pyy\n'
        dispfile += 'variable pyy1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pzz\n'
        dispfile += 'variable pzz1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pxy\n'
        dispfile += 'variable pxy1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pxz\n'
        dispfile += 'variable pxz1 equal ${tmp}\n'
        dispfile += 'variable tmp equal pyz\n'
        dispfile += 'variable pyz1 equal ${tmp}\n'
        dispfile += '\n'
        dispfile += 'variable C1pos equal ${d1}\n'
        dispfile += 'variable C2pos equal ${d2}\n'
        dispfile += 'variable C3pos equal ${d3}\n'
        dispfile += 'variable C4pos equal ${d4}\n'
        dispfile += 'variable C5pos equal ${d5}\n'
        dispfile += 'variable C6pos equal ${d6}\n'
        dispfile += '\n'
        dispfile += 'variable C1${dir} equal 0.5*(${C1neg}+${C1pos})\n'
        dispfile += 'variable C2${dir} equal 0.5*(${C2neg}+${C2pos})\n'
        dispfile += 'variable C3${dir} equal 0.5*(${C3neg}+${C3pos})\n'
        dispfile += 'variable C4${dir} equal 0.5*(${C4neg}+${C4pos})\n'
        dispfile += 'variable C5${dir} equal 0.5*(${C5neg}+${C5pos})\n'
        dispfile += 'variable C6${dir} equal 0.5*(${C6neg}+${C6pos})\n'
        dispfile += '\n'
        dispfile += 'variable dir delete\n'
        return dispfile

    def elastic_in_str() -> str:
        infile = 'include mod.init\n'
        infile += 'include mod.potential\n'
        infile += '\n'
        infile += 'fix 3 all box/relax tri 0.0\n'
        infile += 'minimize ${etol} ${ftol} ${steps} ${fsteps}\n'
        infile += '\n'
        infile += 'variable tmp equal pe\n'
        infile += 'variable e0 equal ${tmp}\n'
        infile += 'variable tmp equal pxx\n'
        infile += 'variable pxx0 equal ${tmp}\n'
        infile += 'variable tmp equal pyy\n'
        infile += 'variable pyy0 equal ${tmp}\n'
        infile += 'variable tmp equal pzz\n'
        infile += 'variable pzz0 equal ${tmp}\n'
        infile += 'variable tmp equal pyz\n'
        infile += 'variable pyz0 equal ${tmp}\n'
        infile += 'variable tmp equal pxz\n'
        infile += 'variable pxz0 equal ${tmp}\n'
        infile += 'variable tmp equal pxy\n'
        infile += 'variable pxy0 equal ${tmp}\n'
        infile += '\n'
        infile += 'variable tmp equal lx\n'
        infile += 'variable lx0 equal ${tmp}\n'
        infile += 'variable tmp equal ly\n'
        infile += 'variable ly0 equal ${tmp}\n'
        infile += 'variable tmp equal lz\n'
        infile += 'variable lz0 equal ${tmp}\n'
        infile += '\n'
        infile += 'variable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}\n'
        infile += 'variable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}\n'
        infile += 'variable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}\n'
        infile += 'variable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}\n'
        infile += 'variable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}\n'
        infile += 'variable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}\n'
        infile += '\n'
        infile += 'unfix 3\n'
        infile += 'write_restart restart.equil\n'
        infile += '\n'
        infile += 'variable dir equal 1\n'
        infile += 'include mod.displace\n'
        infile += '\n'
        infile += 'variable dir equal 2\n'
        infile += 'include mod.displace\n'
        infile += '\n'
        infile += 'variable dir equal 3\n'
        infile += 'include mod.displace\n'
        infile += '\n'
        infile += 'variable dir equal 4\n'
        infile += 'include mod.displace\n'
        infile += '\n'
        infile += 'variable dir equal 5\n'
        infile += 'include mod.displace\n'
        infile += '\n'
        infile += 'variable dir equal 6\n'
        infile += 'include mod.displace\n'
        infile += '\n'
        infile += 'variable C11all equal ${C11}\n'
        infile += 'variable C22all equal ${C22}\n'
        infile += 'variable C33all equal ${C33}\n'
        infile += 'variable C12all equal 0.5*(${C12}+${C21})\n'
        infile += 'variable C13all equal 0.5*(${C13}+${C31})\n'
        infile += 'variable C23all equal 0.5*(${C23}+${C32})\n'
        infile += 'variable C44all equal ${C44}\n'
        infile += 'variable C55all equal ${C55}\n'
        infile += 'variable C66all equal ${C66}\n'
        infile += 'variable C14all equal 0.5*(${C14}+${C41})\n'
        infile += 'variable C15all equal 0.5*(${C15}+${C51})\n'
        infile += 'variable C16all equal 0.5*(${C16}+${C61})\n'
        infile += 'variable C24all equal 0.5*(${C24}+${C42})\n'
        infile += 'variable C25all equal 0.5*(${C25}+${C52})\n'
        infile += 'variable C26all equal 0.5*(${C26}+${C62})\n'
        infile += 'variable C34all equal 0.5*(${C34}+${C43})\n'
        infile += 'variable C35all equal 0.5*(${C35}+${C53})\n'
        infile += 'variable C36all equal 0.5*(${C36}+${C63})\n'
        infile += 'variable C45all equal 0.5*(${C45}+${C54})\n'
        infile += 'variable C46all equal 0.5*(${C46}+${C64})\n'
        infile += 'variable C56all equal 0.5*(${C56}+${C65})\n'
        infile += '\n'
        infile += 'print "========================================="\n'
        infile += 'print "Components of the Elastic Constant Tensor"\n'
        infile += 'print "========================================="\n'
        infile += 'print "Elastic Constant C11all = ${C11all} ${cunits}"\n'
        infile += 'print "Elastic Constant C12all = ${C12all} ${cunits}"\n'
        infile += 'print "Elastic Constant C13all = ${C13all} ${cunits}"\n'
        infile += 'print "Elastic Constant C14all = ${C14all} ${cunits}"\n'
        infile += 'print "Elastic Constant C15all = ${C15all} ${cunits}"\n'
        infile += 'print "Elastic Constant C16all = ${C16all} ${cunits}"\n'
        infile += 'print "Elastic Constant C22all = ${C22all} ${cunits}"\n'
        infile += 'print "Elastic Constant C23all = ${C23all} ${cunits}"\n'
        infile += 'print "Elastic Constant C24all = ${C24all} ${cunits}"\n'
        infile += 'print "Elastic Constant C25all = ${C25all} ${cunits}"\n'
        infile += 'print "Elastic Constant C26all = ${C26all} ${cunits}"\n'
        infile += 'print "Elastic Constant C33all = ${C33all} ${cunits}"\n'
        infile += 'print "Elastic Constant C34all = ${C34all} ${cunits}"\n'
        infile += 'print "Elastic Constant C35all = ${C35all} ${cunits}"\n'
        infile += 'print "Elastic Constant C36all = ${C36all} ${cunits}"\n'
        infile += 'print "Elastic Constant C44all = ${C44all} ${cunits}"\n'
        infile += 'print "Elastic Constant C45all = ${C45all} ${cunits}"\n'
        infile += 'print "Elastic Constant C46all = ${C46all} ${cunits}"\n'
        infile += 'print "Elastic Constant C55all = ${C55all} ${cunits}"\n'
        infile += 'print "Elastic Constant C56all = ${C56all} ${cunits}"\n'
        infile += 'print "Elastic Constant C66all = ${C66all} ${cunits}"\n'
        return infile

    file_map = {
        'mod.init': init_mod_str,
        'mod.potential': potential_mod_str,
        'mod.displace': displace_mod_str,
        'in.elastic': elastic_in_str
    }

    for f in file_map.keys():
        p = Path(f)
        logger.info(f"Write {p} in {os.getcwd()}")
        p.write_text(file_map[f]())


def make_phonon_input(file: Path, potential: Path, minimize_before: bool, thermal: bool, dimensions: list) -> None:
    def phonon_in_str():
        infile = 'units metal\n'
        infile += 'dimension 3\n'
        infile += 'atom_style atomic\n'
        infile += 'atom_modify map array\n'
        infile += 'neighbor 0.3 bin\n'
        infile += 'neigh_modify once no every 1 delay 0 check yes\n'
        infile += 'boundary p p p\n'
        infile += '\n'
        infile += f'read_data {file}\n'
        infile += '\n'
        infile += f'pair_style deepmd {potential}\n'
        infile += 'pair_coeff * *\n'
        return infile
    
    def minimize_in_str():
        infile = 'units metal\n'
        infile += 'dimension 3\n'
        infile += 'atom_style atomic\n'
        infile += 'atom_modify map array\n'
        infile += 'neighbor 0.3 bin\n'
        infile += 'neigh_modify once no every 1 delay 0 check yes\n'
        infile += 'boundary p p p\n'
        infile += '\n'
        infile += f'read_data {file}\n'
        infile += '\n'
        infile += f'pair_style deepmd {potential}\n'
        infile += 'pair_coeff * *\n'
        infile += '\n'
        infile += 'thermo_style custom step fnorm pe pxx pyy pzz pxy pyz pxz\n'
        infile += 'thermo 1\n'
        infile += '\n'
        infile += 'min_style cg\n'
        infile += 'min_modify norm inf\n'
        infile += '\n'
        infile += f'fix f99 all box/relax tri 0.0\n'
        infile += f'minimize 0 1e-10 2000 10000\n'
        infile += 'unfix f99\n'
        infile += '\n'
        infile += 'change_box all x final 0 $(lx) y final 0 $(ly) z final 0 $(lz) remap\n'
        infile += f'write_data {file}\n'
        return infile
    
    def band_conf_str():
        from ase.io.lammpsdata import read_lammps_data
        
        atoms = read_lammps_data(file, style='atomic')
        lat = atoms.cell.get_bravais_lattice()
        bandpath = lat.bandpath()
        
        klabels = bandpath.path.split(',')[0]
        kpts = bandpath.special_points
        
        # Set a different kpath for monoclinic structures
        if klabels == 'GYHCEM1AXH1':
            klabels = 'CYGBAEZ'
            kpts = {
                'A':np.array([-0.5, 0, 0.5]),
                'B': np.array([0, 0, 0.5]),
                'C': np.array([0.5, 0.5, 0]),
                'E': np.array([-0.5, 0.5, 0.5]),
                'G': np.array([0, 0, 0]),
                'Y': np.array([0.5, 0, 0]),
                'Z': np.array([0, 0.5, 0])
                }
        
        band_repr = '   '.join([' '.join([f'{ki}' for ki in kpts[l]]) for l in klabels])
        kpath_repr = ' '.join(klabels)
        kpath_repr = kpath_repr.replace('G', r'$\Gamma$')
        
        infile = 'MP = 20 20 20\n'
        infile += 'WRITE_MESH = .FALSE.\n'
        infile += 'BAND_POINTS = 101\n'
        infile += f'BAND = {band_repr}\n'
        infile += f'BAND_LABELS = {kpath_repr}\n'
        return infile

    def thermal_conf_str():
        infile = 'MP = 20 20 20\n'
        infile += 'WRITE_MESH = .FALSE.\n'
        infile += 'TPROP = .TRUE.\n'
        infile += 'T_MIN = 0\n'
        infile += 'T_MAX = 2000.0\n'
        return infile

    infile = phonon_in_str()
    p = Path('in.phonon')
    logger.info(f"Write {p} in {os.getcwd()}")
    p.write_text(infile)
    
    if minimize_before:
        infile = minimize_in_str()
        p = Path('in.minimize')
        logger.info(f"Write {p} in {os.getcwd()}")
        p.write_text(infile)
        
    infile = band_conf_str()
    p = Path('band.conf')
    logger.info(f"Write {p} in {os.getcwd()}")
    p.write_text(infile)
    
    if thermal:
        infile = thermal_conf_str()
        p = Path('thermal.conf')
        logger.info(f"Write {p} in {os.getcwd()}")
        p.write_text(infile)


def make_neb_input():
    raise NotImplementedError()


def make_thermo_input():
    raise NotImplementedError()


def run_lattice(config):
    root = Path(config['root_dir'])
    pot_path = config['potentials']
    
    lattice_config = config['lattice']
    struct_path = lattice_config['structures']
    
    # Add DFT results if available and requested
    tot_results = ''
    if lattice_config['compare_dft']:
        # TODO implement DFT comparison for lattice
        logger.warning('DFT comparison for lattice computation is not implemented yet')
    
    for pot in pot_path:
        pot_file = pot.name
        
        pot_results = 'Structure     PotEng    Cella      Cellb      Cellc    CellAlpha  CellBeta   CellGamma\n'
        n_separators = (85 - len(pot_file) - 6) // 2
        tot_results += '=' * n_separators + f'   {pot_file}   ' + '=' * n_separators  + '\n'
        
        for struct in struct_path:
            struct_file = struct.name
            phase = struct_file.split('.')[-1]
            
            path = root / pot_file.split('.')[0] / 'LATTICE' / phase
            with change_dir(path):
                logger.debug(f'Enter folder {path}')
                
                # Prepare input files
                get_pot_and_struct(pot, struct)
                make_lattice_input(struct_file, pot_file, **lattice_config['config'])
                
                # Run lammps
                logger.info(f'Launch relaxation of {struct_file}')
                process = subprocess.Popen(['mpirun', '-np', str(AVAIL_CPU), 'lmp', '-in', 'in.lattice'])
                process.wait()
                
                logfile = LogFile('log.lammps')
            
            # Create results string
            pot_results += phase + ' ' * (13 - len(phase))
            for thermo in logfile.run_logs[-1].values():
                value = np.round(thermo, 2)[-1]
                pot_results += str(value) + ' ' * (11 - len(str(value)))
            pot_results += '\n'
        
        tot_results += pot_results + '=' * 85 + '\n\n'
        
        # Write intermediate results in file
        path = root / pot_file.split('.')[0] / 'RESULTS' / 'lattice.dat'
        logger.info(f'Write lattice results for potential {pot_file} in {path}')
        path.write_text(pot_results)
                
    # Write final results in file
    path = root / 'RESULTS' / 'lattice.dat'
    logger.info(f'Write global lattice results in {path}')
    path.write_text(tot_results)
    

def run_elastic(config):
    root = Path(config['root_dir'])
    pot_path = config['potentials']
    
    elastic_config = config['elastic']
    struct_path = elastic_config['structures']
    
    tot_results = ''
    # Add DFT results if available and requested
    if elastic_config['compare_dft']:
        # TODO implement DFT comparison for lattice
        logger.warning('DFT comparison for elastic computation is not implemented yet')
    
    for pot in pot_path:
        pot_file = pot.name
        
        n_separators = (85 - len(pot_file) - 6) // 2
        tot_results += '=' * n_separators + f'   {pot_file}   ' + '=' * n_separators  + '\n'
        pot_results = ''
        
        for struct in struct_path:
            struct_file = struct.name
            phase = struct_file.split('.')[-1]
            
            path = root / pot_file.split('.')[0] / 'ELASTIC' / phase
            with change_dir(path):
                logger.debug(f'Enter folder {path}')
                
                # Prepare input files
                get_pot_and_struct(pot, struct)
                make_elastic_input(struct_file, pot_file, **elastic_config['config'])
                
                # Run lammps
                logger.info(f'Launch elastic tensor calculation of {struct_file}')
                process = subprocess.Popen(['mpirun', '-np', str(AVAIL_CPU), 'lmp', '-in', 'in.elastic'])
                process.wait()
                
                logfile = LogFile('log.lammps')
                elastic = logfile.elastic_tensor
                
            # Create results string
            pot_results += phase + '\n'
            pot_results += '\n'.join([' '.join([f'{x:8.2f}' for x in row]) for row in elastic]) + '\n'
        
        tot_results += pot_results + '=' * 85 + '\n\n'
        
        # # Write intermediate results in file
        path = root / pot_file.split('.')[0] / 'RESULTS' / 'elastic.dat'
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Write lattice results for potential {pot_file} in {path}')
        path.write_text(pot_results)
                
    # Write final results in file
    path = root / 'RESULTS' / 'elastic.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Write global elastic results in {path}')
    path.write_text(tot_results)


def run_phonon(config):
    root = Path(config['root_dir'])
    pot_path = config['potentials']
    
    phonon_config = config['phonon']
    struct_path = phonon_config['structures']
    tot_results = ''
    
    # Add DFT results if available and requested
    if phonon_config['compare_dft']:
        # TODO implement DFT comparison for lattice
        logger.warning('DFT comparison for phonon computation is not implemented yet')
    
    for pot in pot_path:
        pot_file = pot.name
        
        n_separators = (85 - len(pot_file) - 6) // 2
        tot_results += '=' * n_separators + f'   {pot_file}   ' + '=' * n_separators  + '\n'
        pot_results = ''
        
        for struct in struct_path:
            struct_file = struct.name
            phase = struct_file.split('.')[-1]
            
            path = root / pot_file.split('.')[0] / 'PHONON' / phase
            with change_dir(path):
                logger.debug(f'Enter folder {path}')
                
                # Prepare input files
                get_pot_and_struct(pot, struct)
                make_phonon_input(struct_file, pot_file, **phonon_config['config'])
                
                # Run lammps
                logger.info(f'Launch elastic tensor calculation of {struct_file}')
                # process = subprocess.Popen(['mpirun', '-np', str(AVAIL_CPU), 'lmp', '-in', 'in.elastic'])
                # process.wait()
                
    #             logfile = LogFile('log.lammps')
    #             elastic = logfile.elastic_tensor
                
    #         # Create results string
    #         pot_results += phase + '\n'
    #         pot_results += '\n'.join([' '.join([f'{x:8.2f}' for x in row]) for row in elastic]) + '\n'
        
    #     tot_results += pot_results + '=' * 85 + '\n\n'
        
    #     # # Write intermediate results in file
    #     path = root / pot_file.split('.')[0] / 'RESULTS' / 'elastic.dat'
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     logger.info(f'Write lattice results for potential {pot_file} in {path}')
    #     path.write_text(pot_results)
                
    # # Write final results in file
    # path = root / 'RESULTS' / 'elastic.dat'
    # path.parent.mkdir(parents=True, exist_ok=True)
    # logger.info(f'Write global elastic results in {path}')
    # path.write_text(tot_results)

if __name__ == '__main__':
    args = parse_args()
    
    configure_logger(LOGLEVEL)
    config = load_config(args.conf_file)
    
    with change_dir(config['root_dir']):
        config['potentials'] = get_path_from_regex(config['potentials'])

    calculations = {
        'lattice': run_lattice,
        'elastic': run_elastic,
        'phonon': run_phonon,
    }
    
    for calc in calculations:
        if not config[calc]['do']:
            continue
        
        with change_dir(config['root_dir']):
            config[calc]['structures'] = get_path_from_regex(config[calc]['structures'])
        
        calculations[calc](config)
