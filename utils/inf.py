"""
Inference utilities for Numpyro
models.

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
import numpyro as npro
import polars as pl
from jax.typing import ArrayLike


def samples(
    model: Callable,
    cf: dict[str, dict[str, bool | int | float]],
    data: pl.DataFrame,
) -> dict[str, ArrayLike]:
    nuts_kernel = npro.infer.NUTS(
        model,
        dense_mass=True,
        max_tree_depth=cf["inference"]["max_tree_depth"],
        init_strategy=npro.infer.init_to_median,
        target_accept_prob=cf["inference"]["adapt_delta"],
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
        states=jnp.array(data["State"].to_numpy()),
        months=jnp.array(data["Month"].to_numpy()),
        years=jnp.array(data["Year"].to_numpy()),
        tornadoes=jnp.array(data["Tornado"].to_numpy()),
    )
    if cf["inference"]["print_summary"]:
        mcmc.print_summary()
    return mcmc.get_samples()


def prior_predictive_distribution(
    model: Callable, cf: dict[str, dict[str, bool | int | float]]
) -> dict[str, ArrayLike]:
    predictive = npro.infer.Predictive(
        model, num_samples=cf["inference"]["num_samples"]
    )
    rng_key = jr.PRNGKey(cf["reproducibility"]["seed"])
    prior_pred = predictive(rng_key)
    return prior_pred


def posterior_predictive_distribution(
    samples: dict[str, ArrayLike],
    model: Callable,
    cf: dict[str, dict[str, bool | int | float]],
    states: ArrayLike,
    months: ArrayLike,
    years: ArrayLike,
) -> dict[str, ArrayLike]:
    predictive = npro.infer.Predictive(model, posterior_samples=samples)
    rng_key = jr.PRNGKey(cf["reproducibility"]["seed"])
    post_pred = predictive(
        rng_key,
        states=states,
        months=months,
        years=years,
    )
    return post_pred
