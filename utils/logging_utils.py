import json
import logging
import logging.config
import pathlib

log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(save_dir,
                  log_config='conf/logger_config.json',
                  default_level=logging.INFO):
    """Setup logging configuration

    Args:
        save_dir: Path
    """
    log_config = pathlib.Path(log_config)
    if log_config.is_file():
        with open(log_config, 'r') as f:
            config = json.load(f)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(
            log_config))
        logging.basicConfig(level=default_level)


def get_logger(logger_path, logger_name, verbosity=2):
    if not logger_path.exists():
        logger_path.mkdir(parents=True)
    setup_logging(logger_path)
    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
        verbosity, log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_levels[verbosity])
    return logger
