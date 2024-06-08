"""

"""

from typing import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro as npro
import numpyro.distributions as dist
import polars as pl
import scipy
import scipy.stats


def create_model_from_data(data: pl.DataFrame, target_month: int):
    # get data
    clean_data_file_path = "../data/clean/cleaned_NOAA_SPC.csv"
    data = pl.read_csv(clean_data_file_path)

    data_for_target_month = data.filter(pl.col("Month") == target_month)

    tornadoes_by_year = (
        data_for_target_month.select("Year", "Tornado")
        .group_by(["Year"])
        .sum()
    )
    # This is a time series with "Year" and "Tornado"
    # ┌──────┬─────────┐
    # │ Year ┆ Tornado │
    # │ ---  ┆ ---     │
    # │ i64  ┆ i64     │
    # ╞══════╪═════════╡
    # │ 0    ┆ 303     │
    # │ 1    ┆ 351     │
    # │ 2    ┆ 73      │
    # │ 3    ┆ 246     │
    # │ 4    ┆ 121     │
    # └──────┴─────────┘

    # Linear regression, using scipy
    linreg_result = scipy.stats.linregress(
        tornadoes_by_year["Year"], tornadoes_by_year["Tornado"]
    )
    slope = linreg_result.slope
    intercept = linreg_result.intercept

    return LinearRegressionModel(slope, intercept)


class LinearRegressionModel:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def predict_value(self, x: float):
        return x * self.slope + self.intercept


def predictive_model(state=None, month=None, year=None, tornados=None):
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


def inference(
    model: Callable, toml_params: dict, data: pl.DataFrame, save_path: str
):
    nuts_kernel = npro.infer.NUTS(
        model,
        dense_mass=True,
        max_tree_depth=toml_params["inference"]["max_tree_depth"],
        init_strategy=npro.infer.init_to_median,
    )
    mcmc = npro.infer.MCMC(
        nuts_kernel,
        num_warmup=toml_params["inference"]["num_warmup"],
        num_samples=toml_params["inference"]["num_samples"],
        num_chains=toml_params["inference"]["num_chains"],
        progress_bar=toml_params["inference"]["progress_bar"],
    )
    rng_key = jr.PRNGKey(toml_params["reproducibility"]["seed"])
    mcmc.run(
        rng_key,
        state=jnp.array(data["State"].to_numpy()),
        month=jnp.array(data["Month"].to_numpy()),
        year=jnp.array(data["Year"].to_numpy()),
        tornados=jnp.array(data["Tornado"].to_numpy()),
    )
    if toml_params["inference"]["summary"]:
        mcmc.print_summary()
    return mcmc.get_samples()


def posterior_predictive_distribution(samples, model, toml_params):
    predictive = npro.infer.Predictive(model, posterior_samples=samples)
    rng_key = jr.PRNGKey(toml_params["reproducibility"]["seed"])
    post_pred = predictive(
        rng_key,
        state=jnp.array(range(50)),
        month=jnp.array([3]),
        year=jnp.array([2024]),
    )
    return post_pred
