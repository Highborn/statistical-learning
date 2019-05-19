import os


class RuntimeConfig:
    IGNORE_PLOTS = False
    DEBUG_MODE = True
    CONF_PATH = os.path.abspath(__file__)
    CONF_DIR = os.path.dirname(CONF_PATH)
    SRC_DIR = os.path.dirname(CONF_DIR)
    BASE_DIR = os.path.dirname(SRC_DIR)
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    DATA_DIR = os.path.join(STATIC_DIR, 'data')

    PCA_DEFAULT_STANDARD = True
    PCA_DEFAULT_VARIANCE_THRESHOLD = .900

    LDA_DEFAULT_STANDARD = True
    LDA_DEFAULT_VARIANCE_THRESHOLD = .995
