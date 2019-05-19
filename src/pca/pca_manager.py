import numpy as np
from sklearn.preprocessing import StandardScaler

from src.common.singleton_meta_class import Singleton
from src.config.runtime_config import RuntimeConfig
from src.discriminant_analysis.discriminant_analysis_manager import DiscriminantAnalysisManager
from src.pca.pca_dal import PCADal


class PCAManager(DiscriminantAnalysisManager, metaclass=Singleton):
    def __init__(self, data_path=None, threshold=None, standard=True):
        if standard is None:
            standard = RuntimeConfig.PCA_DEFAULT_STANDARD
        if threshold is None:
            threshold = RuntimeConfig.PCA_DEFAULT_VARIANCE_THRESHOLD

        self.dal = PCADal(data_path)
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
        covariacne_matrix = self.compute_covariance_matrix(feature_matrix)
        ordered_eigenvalues, ordered_eigenvectors = self.compute_eigens(covariacne_matrix)
        projection_factor = self.compute_projection_factor(ordered_eigenvalues, ordered_eigenvectors)
        projected_feature_matrix = self.compute_new_feature_matrix(label_array, feature_matrix, projection_factor)
        return projected_feature_matrix

    def compute_covariance_matrix(self, feature_matrix):
        mean_vector = np.mean(feature_matrix, axis=0)
        instance_count = feature_matrix.shape[0]
        least_square_exp = feature_matrix - mean_vector
        covariacne_matrix = least_square_exp.T.dot(least_square_exp) / (instance_count - 1)
        return covariacne_matrix


if __name__ == '__main__':
    manager = PCAManager()
    manager.solve()
    print('hello, world')
