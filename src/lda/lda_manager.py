import numpy as np

from src.common.singleton_meta_class import Singleton
from src.config.runtime_config import RuntimeConfig
from src.discriminant_analysis.discriminant_analysis_manager import DiscriminantAnalysisManager
from src.lda.lda_dal import LDADal
from sklearn.preprocessing import StandardScaler


class LDAManager(DiscriminantAnalysisManager, metaclass=Singleton):
    def __init__(self, data_path=None, threshold=None, standard=True):
        if standard is None:
            standard = RuntimeConfig.LDA_DEFAULT_STANDARD
        if threshold is None:
            threshold = RuntimeConfig.LDA_DEFAULT_VARIANCE_THRESHOLD
        self.dal = LDADal(data_path)
        self.standard = standard
        self.threshold = threshold

    def _populate(self, label_array):
        self.label_items = np.unique(label_array)
        self.label_count = len(self.label_items)
        self.feature_count = self.dal.feature_count

    def solve(self):
        label_array, feature_matrix = self.dal.read_data()
        return self.project_features(feature_matrix, label_array)

    def project_features(self, label_array, feature_matrix):
        self._populate(label_array)
        feature_matrix = StandardScaler().fit_transform(feature_matrix)
        mean_matrix = np.zeros((self.label_count, self.feature_count))
        for idx, label in enumerate(self.label_items):
            evidence_list = feature_matrix[label_array == label]
            mean_matrix[idx] = np.mean(evidence_list, axis=0)
        within_scatter_matrix = self.compute_within_scatter_matrix(feature_matrix, label_array, mean_matrix)
        between_scatter_matrix = self.compute_between_scatter_matrix(feature_matrix, label_array, mean_matrix)
        multiply = np.linalg.inv(within_scatter_matrix).dot(between_scatter_matrix)
        ordered_eigenvalues, ordered_eigenvectors = self.compute_eigens(multiply)
        projection_factor = self.compute_projection_factor(ordered_eigenvalues, ordered_eigenvectors)
        projected_feature_matrix = self.compute_new_feature_matrix(label_array, feature_matrix, projection_factor)
        return projected_feature_matrix

    def compute_between_scatter_matrix(self, feature_matrix, label_array, mean_matrix):
        between_scatter_matrix = np.zeros((self.feature_count, self.feature_count))
        total_mean_vector = np.mean(feature_matrix, axis=0).reshape(self.feature_count, 1)
        for idx in range(mean_matrix.shape[0]):
            n = feature_matrix[label_array == label_array[idx]].shape[0]
            mean_vector = mean_matrix[idx:idx + 1].T
            between_scatter_matrix += n * (mean_vector - total_mean_vector).dot((mean_vector - total_mean_vector).T)
        return between_scatter_matrix

    def compute_within_scatter_matrix(self, feature_matrix, label_array, mean_matrix):
        within_scatter_matrix = np.zeros((self.feature_count, self.feature_count))
        for idx, label in enumerate(self.label_items):
            scatter_matrix = np.zeros((self.feature_count, self.feature_count))
            evidence_list = feature_matrix[label_array == label]
            mean_vector = mean_matrix[idx:idx + 1].T
            for i in range(evidence_list.shape[0]):
                evidence = evidence_list[i:i + 1].T
                scatter_matrix += (evidence - mean_vector).dot((evidence - mean_vector).T)
            within_scatter_matrix += scatter_matrix
        return within_scatter_matrix
