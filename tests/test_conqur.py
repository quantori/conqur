#!/usr/bin/env python

"""Tests for `conqur` package."""

import unittest
import pytest
import numpy as np
import pandas as pd
import math
from click.testing import CliRunner

from conqur import scaler
from conqur import cli


#@pytest.fixture
def r_initial_matrix():
    with open('C:/Users/a.grefenshteyn/Desktop/tests/initial_matrix_for_test.txt') as file:
        X_batchid = np.array(pd.read_csv(file, sep=' '))
        X_with_batch_columns = np.zeros((273, 107))
        X_with_batch_columns[:, range(100)] = X_batchid[:, range(100)]
        X_with_batch_columns[:, range(103, 107)] = X_batchid[:, range(101, 105)]
        for j in range(273):
            if X_batchid[j][100] == 0:
                X_with_batch_columns[j][100] = 1
            elif X_batchid[j][100] == 1:
                X_with_batch_columns[j][101] = 1
            else:
                X_with_batch_columns[j][102] = 1
        return X_with_batch_columns


#@pytest.fixture
def r_corrected_matrix_1():
    with open('C:/Users/a.grefenshteyn/Desktop/tests/matrix_corrected_for_test_1.txt') as file:
        return np.array(pd.read_csv(file, sep=' '))


#@pytest.fixture
#def r_corrected_matrix_2():
#    with open('C:/Users/a.grefenshteyn/Desktop/tests/matrix_corrected_for_test_2.txt') as file:
#        return np.array(pd.read_csv(file, sep=' '), dtype=np.int16)


class TestConqur(unittest.TestCase):
    """Tests for `conqur` package."""

    def test_from_R_data_1(self):
        ConQur_class = scaler.ConQur(np.array([100, 101, 102]),
                                     np.array([103, 104, 105, 106]),
                                     {100: 1, 101: 0, 102: 0},
                                     penalty='none',
                                     alphas=np.zeros(100))
        Xt = ConQur_class.fit_transform(r_initial_matrix())
        assert np.array_equal(Xt[:, 0], r_corrected_matrix_1()[:, 0])

#    def test_from_R_data_2(self):
#        ConQur_class = scaler.ConQur(np.array([101]), np.array([102, 103, 104, 105]), penalty='l1',
#                                     alpha=math.pi * 4 / 3)
#        Xt = ConQur_class.fit_transform(r_initial_matrix())
#        assert Xt == r_corrected_matrix_2()

#    def setUp(self):
#       """Set up test fixtures, if any."""
#
#    def tearDown(self):
#       """Tear down test fixtures, if any."""
#
#   def test_000_something(self):
#      """Test something."""
#
#   def test_command_line_interface(self):
#      """Test the CLI."""
#        runner = CliRunner()
#        result = runner.invoke(cli.main)
#        assert result.exit_code == 0
#        assert 'conqur.cli.main' in result.output
#        help_result = runner.invoke(cli.main, ['--help'])
#        assert help_result.exit_code == 0
#        assert '--help  Show this message and exit.' in help_result.output
