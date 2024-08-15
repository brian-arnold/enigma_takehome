"""
This implements an end-to-end test to check for any changes in behavior throughout 
the entire factor analysis with EM. The previous state of the software results were
recorded, and we check to see if the results diverge after a new run of
the the entire EM algorithm.

We also verify some of the core assumptions of the model:
1.) the off-diagonal elements of Psi should be 0
2.) the latent factors should follow a normal distribution

A previous run of the code was under the following conditions: 
    1.) random seed = 0
    2.) number of samples = 1000
    3.) number of features = 2
    4.) covariance matrix with diagonal entries = 1, off-diagonal entries = 0.5
    4.) number of latent factors = 1
"""

import pytest
import numpy as np
np.random.seed(0) # random seed needs to be 0
from my_package import functions
from scipy.stats import shapiro

@pytest.fixture
def data():
    # Create a synthetic dataset
    samples = 1000
    features = 2
    var, covar = 1, 0.5

    mu = np.zeros(features)
    # create covariance matrix, diag = 1, off-diag = 0.5
    sigma = np.full((features, features), covar)
    np.fill_diagonal(sigma, var)

    data = np.random.multivariate_normal(mu, sigma, samples)
    data = data.T

    return data

def test_em_factor_analysis(data):
    latent = 1
    features = data.shape[0]

    # Randomly initialize Lambda and Psi
    Lambda = np.random.randn(features, latent)  # Factor loading matrix for one factor
    Psi = np.diag(np.random.rand(features))  # Specific variance matrix (diagonal)

    Lambda, Psi, _, _ = functions.em_factor_analysis(data, Lambda, Psi, tol=0.000001, max_iter=1000)

    # compute discrepancy between original covariance and model covariance
    # ensure it's similar to a previously recorded state
    original_covariance = np.cov(data)
    model_covariance = Lambda @ Lambda.T + Psi
    error_new = np.linalg.norm(original_covariance - model_covariance, 'fro')
    error_prev = 0.0008741496341266082 # this number was taken from a previous run of the function
    assert np.allclose(error_new, error_prev, atol=1e-5)

    # testing some other critical assumptions:
    # the off diagonal elements of Psi should be precisely 0 based on the code
    indices = np.where(~np.eye(Psi.shape[0],dtype=bool))
    assert np.all(Psi[indices] == 0)

    # the latent factors should follow a normal distribution
    E_z, _ = functions.E_step(data, Lambda, Psi)
    _, p = shapiro(E_z[0,:])
    assert p >= 0.05
    
