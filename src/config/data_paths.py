import os

from src.config.runtime_config import RuntimeConfig


class DefaultData:
    ROOT_PATH = None
    TEST_DATA = None
    TRAIN_DATA = None

    LABELED = None
    FEATURE_COUNT = None
    CLASS_COUNT = None


class TitanicData(DefaultData):
    ROOT_PATH = os.path.join(RuntimeConfig.DATA_DIR, 'titanic')
    TEST_DATA = os.path.join(ROOT_PATH, 'train.csv')
    TRAIN_DATA = os.path.join(ROOT_PATH, 'test.csv')
    NAMES_PATH = os.path.join(ROOT_PATH, 'iris.names')

    LABELED = True
    FEATURE_COUNT = 11
    CLASS_COUNT = 2


class IRISData(DefaultData):
    ROOT_PATH = os.path.join(RuntimeConfig.DATA_DIR, 'iris')
    TRAIN_DATA = os.path.join(ROOT_PATH, 'iris.data')
    NAMES_PATH = os.path.join(ROOT_PATH, 'iris.names')

    LABELED = True
    FEATURE_COUNT = 4
    CLASS_COUNT = 3
