"""
Tornado model E414 01. 
This model includes variations
on an underlying multilevel model.

Authors
-------
118777

Date Created
------------
2024-04-23

Last Updated
------------
2024-04-28
"""

from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import polars as pl


def model_01(
    states=None, 
    months=None, 
    years=None, 
    tornados=None):

    num_states = len(np.unique(states))
    num_months = len(np.unique(months))
    num_years = len(np.unique(years))

    sigma_alpha = npro.sample("sigma_alpha", dist.Exponential(1.0))
    sigma_gamma = npro.sample("sigma_gamma", dist.Exponential(1.0))
    sigma_delta = npro.sample("sigma_delta", dist.Exponential(1.0))
    sigma_theta = npro.sample("sigma_theta", dist.Exponential(1.0))

    alpha = npro.sample(
        "alpha", dist.Normal(0, sigma_alpha), sample_shape=(num_states,)
    )
    gamma = npro.sample(
        "gamma", dist.Normal(0, sigma_gamma), sample_shape=(num_months,)
    )
    delta = npro.sample(
        "delta", dist.Normal(0, sigma_delta), sample_shape=(num_years,)
    )
    theta = npro.sample(
        "theta",
        dist.Normal(0, sigma_theta),
        sample_shape=(num_months, num_years),
    )

    lambda_ = jnp.exp(
        alpha[states] + gamma[months] + delta[years] + theta[months, years]
    )
    with npro.plate("data", size=len(tornados)):
        npro.sample("obs", dist.Poisson(lambda_), obs=tornados)