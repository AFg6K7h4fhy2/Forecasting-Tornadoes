""" 
File nearing completion.
"""

import toml 
import os 
from copy import copy
from toml.decoder import TomlDecodeError


def load_and_valid_config(
    cf_path: str
) -> dict[str, str | bool | int]:
    """
    Loads and validations model params.toml configuration files

    Parameters
    ----------
    cf_path : str
        The configuration file path

    Returns
    -------
    dict
        A dictionary of config params.

    Notes
    -----
    Access the configuration parameters in the following manner
    cf["<category>"]["<category_config_param"]
    """
    
    # verify files exists 
    if not os.path.exists(cf_path):
        raise ValueError(f"{cf_path} does not point to a valid configuration file")

    # try to load the configuration file
    try: 
        with open(cf_path, "r") as toml_file:
            cf = toml.load(toml_file)
            toml_file.close()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The configuration file was not found at {cf_path}.")
    except TomlDecodeError as e:
        raise TomlDecodeError(
            f"Error decoding the TOML file: {e}")
    except Exception as e: 
        raise Exception(
            f"{e}\nThe toml configuration file was unable to be loaded."
        )

    # # make sure all inputted primary keys are supported
    valid_key_combinations = {
        "reproducibility": {"seed"}, 
        "data": {"preliminary", "start_year", "forecast_month", "forecast_state", "forecast_year"}, 
        "tornado_parameters": {}, 
        "inference": {"adapt_delta", "max_tree_depth", "num_warmup", "num_samples", "num_chains", "progress_bar", "print_summary", "print_post_prior", "save_obs_summary", "save_samples", "save_prior_pred", "save_post_pred"}, 
        "output": {"report", "show_map_plot", "show_bar_plots", "show_bar_plots", "show_mcmc_trace", "show_post_pred", "show_prior_pred", "show_bayes_plot"}}
    for primary_key, secondary_keys in cf.items():
        if primary_key not in valid_key_combinations:
            raise ValueError(
                f"The category (primary key) you had ({primary_key}) in your toml file is invalid"
            )
        for secondary_key in secondary_keys:
            if secondary_key not in valid_key_combinations[primary_key]:
                raise ValueError(
                    f"The secondary key ({secondary_key}) you entered in section {primary_key} is not a valid option"
                )

    # make sure necessary keys combinations are present 
    necessary_key_combinations = {
        "reproducibility": {"seed"}, 
        "data": {"start_year", "forecast_month", "forecast_state", "forecast_year"},
        "inference": {"num_samples", "num_warmup"}
    }
    set_necessary = set(
        [(k, sk) for k, v in necessary_key_combinations.items() for sk in v])
    set_cf = set(
        [(k, sk) for k, v in cf.items() for sk in v.keys()])
    if not set_necessary.issubset(set_cf):
        raise ValueError("The following primary keys and their settings are not present in the configuration file you entered, but are necessary :\n{necessary_key_combinations}")

    # fix non-necessary keys to defaults if not entered 
    optional_key_defaults = {
        "data": {"preliminary": True},
        "inference": {
            "adapt_delta": 0.85, 
            "max_tree_depth": 10, 
            "num_warmup": 100, 
            "num_samples": 250, 
            "num_chains": 1, 
            "progress_bar": True, 
            "print_summary": True, 
            "print_post_prior": True,
            "save_obs_summary": False, 
            "save_samples": False, 
            "save_prior_pred": False, 
            "save_post_pred": False}, 
        "output": {
            "report": False, 
            "show_map_plot": True,
            "show_bar_plots": False, 
            "show_mcmc_trace": True, 
            "show_post_pred": True, 
            "show_prior_pred": True, 
            "show_bayes_plot": True}
    }

    # make sure all necessary keys are inputted
    for primary_key, secondary_keys in optional_key_defaults.items():
        if primary_key not in cf:
            cf[primary_key] = copy(optional_key_defaults[primary_key])
        else:
            for secondary_key in secondary_keys:
                if secondary_key not in cf[primary_key]:
                    cf[primary_key][secondary_key] = copy(optional_key_defaults[primary_key][secondary_key])
    
    # verify that config parameter values are appropriate and correctly typed
    assert isinstance(cf["reproducibility"]["seed"], int),""
    assert 1<=cf["reproducibility"]["seed"]<=1000000
    assert isinstance(cf["data"]["preliminary"], bool),"Preliminary must be of type boolean"
    assert isinstance(cf["data"]["start_year"], int),"The start year must be any integer."
    assert 1<=cf["data"]["start_year"],"The start year must be positive."


    assert isinstance(cf["data"]["year"], int),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""
    assert isinstance(cf[""][""], ),""


# CHECK that preliminary is boolean
# CHECK that files in raw contain selected preliminary
# CHECK that start year is a positive integer
# CHECK that start year is at least min year, at most max year
# CHECK that forecast_month is str
# CHECK forecast_month = "April" in months
# CHECK that forecast_state is str
# CHECK that forecast_state in us_state_abbreviations + ["US"]
# CHECK that forecast_year is positive integer 
# CHECK that forecast_year is at least 1 + min year, at most 5 + max_year


# [tornado_parameters]


# [inference]
# adapt_delta = 0.85
# max_tree_depth = 12
# num_warmup = 50
# num_samples = 400
# num_chains = 1
# progress_bar = true
# print_summary = true
# save_obs_summary = true
# save_samples = true
# save_prior_pred = true
# save_post_pred = true
# 
# CHECK that adapt_delta is float 
# CHECK that adapt_delta is between [0.6, 0.95]
# CHECK that max_tree_depth is int 
# CHECK that max_tree_depth is between [2, 12]
# CHECK that num_warmup is positive int
# CHECK that num_warmup is between 50 and 1000
# CHECK that num_samples is positive int
# CHECK that num_samples is between 100 and 2500
# CHECK that progress_bar, summary, save_samples

# [visuals]
# report = true
# show_data_plots = true
#   US map w/ coloration (each year)
#   state bar w/ counts (each year)
#   month bar w/ counts by (each state, year)
# show_mcmc_trace = true
#   sample trace
#   smooth distribution
#   histogram
# show_post_pred = true
# show_prior_pred = true
# show_bayes_plot = true

# 


load_and_valid_config("../data/params_A.toml")
    
