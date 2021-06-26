import numpy as np
from numpy.testing import assert_array_equal
import pytest
from pystmm.classifier import (STMB, STMM)

def test_STMB_fit():
    """Test Fit of STMB"""
    clf = STMB(C1=1.0, C2=1.0, maxIter=30, tolSTM=1e-4, penalty = 'l2', dual = True, tol=1e-4,loss = 'squared_hinge', maxIterSVM=100000)

def test_STMM_fit():
    """Test Fit of STMB"""
    clf = STMB(typemulticlassifier='ovr',C1=1.0, C2=1.0, maxIter=30, tolSTM=1e-4, penalty = 'l2', dual = True, tol=1e-4,loss = 'squared_hinge', maxIterSVM=100000)