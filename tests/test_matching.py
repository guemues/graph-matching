import matching
import unittest
import numpy as np
import pandas as pd

from matching.utils import mapping_dataframe


class TestStringMethods(unittest.TestCase):

    def test_match_using_threshold(self):
        result = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
        distance_matrix = ((np.arange(16).reshape((4, 4)) + np.arange(16).reshape((4, 4)).T) / 2)
        result_2 = matching.match_using_threshold(distance_matrix, 0.5)

        assert np.allclose(result, result_2)

    def test_node_information(self):
        distances = np.array([[.1, .3, .4, .5], [.3, .2, .7, .4], [.4, .7, .1, .3], [.5, .4, .3, .4]])
        mapping_1 = {0: 0, 1: 1, 2: 2, 3: 3}
        mapping_2 = {0: 0, 1: 1, 2: 2, 3: 3}

        _ = mapping_dataframe(distances, mapping_1, mapping_2, .1, .5)

        assert _['node_1'].equals(pd.Series([0, 0, 1, 1, 2, 2, 3]))
        assert _['node_2'].equals(pd.Series([0, 1, 0, 1, 2, 3, 2]))

