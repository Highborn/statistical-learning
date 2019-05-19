import random
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.common.singleton_meta_class import Singleton
from src.config.runtime_config import RuntimeConfig


class DiscriminantAnalysisManager(metaclass=Singleton):
    threshold = None
    feature_count = None
    label_items = None
    label_count = None

    def compute_eigens(self, matrix) -> Tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        ordered_eigenvalues = eigenvalues[order]
        ordered_eigenvectors = eigenvectors[order]
        return ordered_eigenvalues, ordered_eigenvectors

    def compute_projection_factor(self, ordered_eigenvalues, ordered_eigenvectors):
        total = sum(ordered_eigenvalues)
        explained_variances = [(eigenvalue / total) for eigenvalue in ordered_eigenvalues]
        aggregation_explanations = np.cumsum(explained_variances)

        self.plot_explained_variances(explained_variances, aggregation_explanations)

        d = 0
        satisfied = False
        for variance in aggregation_explanations:
            d += 1
            if variance > self.threshold:
                satisfied = True
                break
        if not satisfied:
            raise ValueError("There is no solution!")

        projection_factor = np.hstack(
            (eigenvector.reshape(self.feature_count, 1) for eigenvector in ordered_eigenvectors.T[0:d])
        )
        return projection_factor

    def plot_explained_variances(self, explained_variances, aggregation_explanations):
        if RuntimeConfig.IGNORE_PLOTS:
            return
        n_groups = self.feature_count

        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        plt.bar(
            index,
            explained_variances,
            bar_width,
            alpha=opacity, color='#700c8e', label='Variance Explanation',
        )
        plt.plot(
            index,
            list(aggregation_explanations),
            bar_width,
            alpha=opacity, color='orange', label='Cumulative Variance Explanation',
        )

        for idx, value in enumerate(explained_variances):
            ax.text(idx, value + 0.05, str(value)[:5], va='center', color='#700c8e', fontweight='bold')
        for idx, value in enumerate(aggregation_explanations):
            ax.text(idx, value - 0.05, str(value)[:5], va='center', color='orange', fontweight='bold')

        plt.xlabel('Features')
        plt.ylabel('Explained Variation')
        plt.title('Variations by Features')
        labels = (str(x) for x in range(n_groups))
        plt.xticks(index, labels)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def compute_new_feature_matrix(self, label_array, feature_matrix, projection_factor):
        projected_feature_matrix = feature_matrix.dot(projection_factor)
        self.plot_result(label_array, projected_feature_matrix)

        return projected_feature_matrix

    def get_random_color(self):
        return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

    def plot_result(self, label_array, projected_feature_matrix):
        if RuntimeConfig.IGNORE_PLOTS:
            return

        color_list = ['r', 'g', 'b']
        color_list.extend([self.get_random_color() for _ in range(self.label_count)])
        color_list = color_list[:self.label_count]
        for idx in range(self.label_count):
            plt.plot(
                projected_feature_matrix[label_array == self.label_items[idx]][:, 0],
                projected_feature_matrix[label_array == self.label_items[idx]][:, 1],
                'o', markersize=7, color=color_list[idx], alpha=0.5, label='class ' + str(idx),
            )

        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.legend()
        plt.title('Transformed samples with class labels')
        plt.show()
