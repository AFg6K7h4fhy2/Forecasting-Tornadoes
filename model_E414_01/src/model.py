"""
Tornado model E414 01. This
file consists of several models
built off of the E414 01 model
iteration, which consist of
several related multilevel
models for tornado counts.

Authors
-------
118777

Date Created
------------
2024-04-23

Last Updated
------------
2024-04-29
"""

import jax.numpy as jnp
import numpy as np
import numpyro as npro
import numpyro.distributions as dist


def model_01(states=None, months=None, years=None, tornadoes=None):

    # states, months, years, and tornadoes represented
    num_states = len(np.unique(states))
    num_months = len(np.unique(months))
    num_years = len(np.unique(years))

    # state effect hyperparameters
    alpha_mu = npro.sample("alpha_mu", dist.Normal(0, 3.0))
    alpha_sigma = npro.sample("alpha_sigma", dist.Exponential(1.0))

    # state effect
    with npro.plate("states", num_states):
        alpha = npro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))

    # month effect hyperparameters
    gamma_mu = npro.sample("gamma_mu", dist.Normal(0, 1.0))
    gamma_sigma = npro.sample("gamma_sigma", dist.Exponential(1.0))

    # month effect
    with npro.plate("months", num_months):
        gamma = npro.sample("gamma", dist.Normal(gamma_mu, gamma_sigma))

    # year effect hyperparameters
    delta_mu = npro.sample("delta_mu", dist.Normal(0, 2.0))
    delta_sigma = npro.sample("delta_sigma", dist.Exponential(1.0))

    # year effect
    with npro.plate("years", num_years):
        delta = npro.sample("delta", dist.Normal(delta_mu, delta_sigma))

    # expected tornadoes
    Y = jnp.exp(alpha[states] + gamma[months] + delta[years])

    # likelihood
    # with npro.plate("data", size=len(tornadoes)):
    npro.sample("obs", dist.Poisson(Y), obs=tornadoes)


def model_02(states=None, months=None, years=None, tornados=None):

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
