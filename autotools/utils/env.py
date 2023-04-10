import os
import logging

logger = logging.getLogger(__name__)

AVAILABLE_ENV = ['OMP_NUM_THREADS',
                 'TF_INTER_OP_PARALLELISM_THREADS',
                 'TF_INTRA_OP_PARALLELISM_THREADS',
                 ]


def _set_env_with_default(key: str, value: str, default: str):
    if os.environ.get(key) is None:
        os.environ[key] = default
        logger.info(
            f"Environment variable {key} is empty. Use the default value {default}")
    else:
        os.environ[key] = str(config[key])
        logger.debug(f"Environment variable {key} is set to {os.environ[key]}")


def set_env(cfg):
    for key in AVAILABLE_ENV:
        _set_env_with_default(key, cfg[key], '1')
    _set_env_with_default('OMP_NUM_THREADS',
                          cfg['OMP_NUM_THREADS'], '1')

    _set_env_with_default('TF_INTER_OP_PARALLELISM_THREADS',
                          cfg['TF_INTER_OP_PARALLELISM_THREADS'], '0')

    _set_env_with_default('TF_INTRA_OP_PARALLELISM_THREADS',
                          cfg['TF_INTRA_OP_PARALLELISM_THREADS'], '0')
