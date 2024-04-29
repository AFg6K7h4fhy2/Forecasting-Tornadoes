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

import jax.numpy as jnp
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
    samples = ut_inf.samples(model_01, cf, data)

    # get posterior samples
    post_samples = ut_inf.posterior_predictive_distribution(
        samples,
        model_01,
        cf,
        states=jnp.array(list(range(1, 50 + 1))),
        months=jnp.array([4]),
        years=jnp.array([5]),
    )

    print(post_samples)
    print(post_samples["obs"])
    print(jnp.mean(post_samples["obs"], axis=1))
    print(jnp.sum(jnp.mean(post_samples["obs"], axis=1)))
    print(jnp.mean(post_samples["obs"], axis=0))
    print(jnp.sum(jnp.mean(post_samples["obs"], axis=0)))


if __name__ == "__main__":
    main()
