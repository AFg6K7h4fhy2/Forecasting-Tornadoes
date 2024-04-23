"""
File in need of some changes.
Add prior predictive.
Improve posterior predictive.
"""

from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import polars as pl


def tornado_modelA(state=None, month=None, year=None, tornados=None):
    num_states = len(np.unique(state))
    num_months = len(np.unique(month))
    num_years = len(np.unique(year))

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
        alpha[state] + gamma[month] + delta[year] + theta[month, year]
    )
    # with npro.plate("data", size=len(tornados)):
    npro.sample("obs", dist.Poisson(lambda_), obs=tornados)


def inference(model: Callable, cf: dict, data: pl.DataFrame, save_path: str):
    nuts_kernel = npro.infer.NUTS(
        model,
        dense_mass=True,
        max_tree_depth=cf["inference"]["max_tree_depth"],
        init_strategy=npro.infer.init_to_median,
    )
    mcmc = npro.infer.MCMC(
        nuts_kernel,
        num_warmup=cf["inference"]["num_warmup"],
        num_samples=cf["inference"]["num_samples"],
        num_chains=cf["inference"]["num_chains"],
        progress_bar=cf["inference"]["progress_bar"],
    )
    rng_key = jr.PRNGKey(cf["reproducibility"]["seed"])
    mcmc.run(
        rng_key,
        state=jnp.array(data["State"].to_numpy()),
        month=jnp.array(data["Month"].to_numpy()),
        year=jnp.array(data["Year"].to_numpy()),
        tornados=jnp.array(data["Tornado"].to_numpy()),
    )
    if cf["inference"]["summary"]:
        mcmc.print_summary()
    return mcmc.get_samples()


def posterior_predictive_distribution(samples, model, cf):
    predictive = npro.infer.Predictive(model, posterior_samples=samples)
    rng_key = jr.PRNGKey(cf["reproducibility"]["seed"])
    post_pred = predictive(
        rng_key,
        state=jnp.array(list(range(50))),
        month=jnp.array([3]),
        year=jnp.array([2024]),
    )
    return post_pred
