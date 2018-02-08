import os
import yaml

from commons.logger import logger


def read():
    # Assume the file is located in commons
    # Assume that the file {root-project}/conf/conf.yml exists
    #
    # {root-project}
    # - conf/
    #   - conf.yml
    #   - ...
    # - commons/
    #   - configuration.py
    #   - ...
    SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(SCRIPT_DIR, 'conf/api-conf.yml')
    logger.info("Using conf file: {}".format(path))
    return yaml.load(open(path))


def load():
    return read()
