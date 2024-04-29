"""
Code for visualizing raw tornado data counts
and activity across time by state. Can also
be used to visualize prior and posterior
predictive checks, inference results, and
forecasts.

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

import numpyro as npro


def plot_prior_predictive():
    pass


def plot_posterior_predictive():
    pass


def plot_mcmc_samples():
    pass


def plot_posterior_samples():
    pass


def plot_priors(
    dist: npro.distributions,
    num_samples: int,
    is_hyper: bool,
    style_path: str,
    model_name: str,
    save_path: str,
):
    pass


def run_plot_priors():
    """
    A manually useable function for
    plotting prior distributions and
    generating figures from them.
    """
    pass
