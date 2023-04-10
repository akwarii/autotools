from dataclasses import dataclass


@dataclass
class Env:
    omp_num_threads: int | None
    tf_inter_op_parallelism_threads: int | None
    tf_intra_op_parallelism_threads: int | None


@dataclass
class Slurm:
    job_name: str
    partition: list[str]
    time: str
    nodes: int
    ntasks: int


@dataclass
class DPSettings:
    steps: int
    ftol: float


@dataclass
class DPCalculator:
    use: bool
    models: list[str]
    structures: list[str]
    compare_with: str | None
    settings: DPSettings


@dataclass
class VaspSettings:
    relax_incar: str | None
    static_incar: str | None
    potcar: str
    kpoints: str | None


@dataclass
class VaspCalculator:
    use: bool
    structures: list[str]
    compare_with: str | None
    settings: VaspSettings
    command: str


@dataclass
class Calculators:
    dp: DPCalculator
    vasp: VaspCalculator


@dataclass
class Relax:
    run: bool
    relax_box: bool


@dataclass
class Lattice:
    run: bool


@dataclass
class EOS:
    run: bool
    compare_with: str
    points: int
    deformation: float


@dataclass
class Elastic:
    run: bool
    compare_with: str
    deformation: float


@dataclass
class Phonon:
    run: bool
    compare_with: str
    dimensions: list[int]
    displacement: float
    kpath: str | None


@dataclass
class Calculation:
    relax: Relax
    lattice: Lattice
    eos: EOS
    elastic: Elastic
    phonon: Phonon


@dataclass
class AutoToolsConfig:
    env: Env
    slurm: Slurm
    calculators: Calculators
    calculation: Calculation
