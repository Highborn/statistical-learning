from typing import Tuple

from numpy import ndarray
from pandas import read_csv

from src.config.data_paths import IRISData


class DiscriminantAnalysisDal:
    def __init__(self, data_path=None, feature_count=None, class_count=None, labeled=None):
        if data_path is None:
            data_path = IRISData.TRAIN_DATA
        if feature_count is None:
            feature_count = IRISData.FEATURE_COUNT
        if class_count is None:
            class_count = IRISData.CLASS_COUNT
        if labeled is None:
            labeled = IRISData.LABELED
        self.data_path = data_path
        self.feature_count = feature_count
        self.class_count = class_count
        self.labeled = labeled

    def read_data(self) -> Tuple[ndarray, ndarray]:
        data_file = read_csv(
            filepath_or_buffer=self.data_path,
            header=None,
            sep=',',
        )

        data_file.dropna(how="all", inplace=True)
        label_array = data_file.iloc[:, self.feature_count].values
        feature_matrix = data_file.iloc[:, 0:self.feature_count].values
        return label_array, feature_matrix
