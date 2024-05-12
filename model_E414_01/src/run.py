"""
Run model(s) E414 01.

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

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
from model import model_01

import utils.config as ut_config
import utils.inf as ut_inf


def main():

    # load config
    cf = ut_config.load_and_valid_config("../params.toml")

    # load data
    data_path = "../../data/clean/cleaned_NOAA_SPC.csv"
    data = pl.read_csv(data_path)

    # get samples
    samples, mcmc = ut_inf.samples(model_01, cf, data)

    # prior predictive
    prior_pred = ut_inf.prior_predictive_distribution(model_01, cf)
    print(prior_pred["obs"])

    # posterior predictive check
    target_states = jnp.array(data["State"].to_numpy())
    target_months = jnp.array(data["Month"].to_numpy())
    target_years = jnp.array(data["Year"].to_numpy())
    post_pred_check = ut_inf.posterior_predictive_distribution(
        samples, model_01, cf, target_states, target_months, target_years
    )
    print(post_pred_check["obs"])

    # posterior predictive forecasting
    target_states = jnp.array(data["State"].to_numpy())
    target_months = jnp.array([4])
    target_years = jnp.array(data["Year"].to_numpy())
    post_pred_forecast = ut_inf.posterior_predictive_distribution(
        samples, model_01, cf, target_states, target_months, target_years
    )
    print(post_pred_forecast["obs"])
    print(jnp.mean(post_pred_forecast["obs"], axis=1))
    print(jnp.sum(jnp.mean(post_pred_forecast["obs"], axis=1)))

    az.plot_ppc(az.from_numpyro(mcmc, posterior_predictive=post_pred_forecast))
    plt.show()

# use symlinks
# .gitignore it after
# 

if __name__ == "__main__":
    main()
