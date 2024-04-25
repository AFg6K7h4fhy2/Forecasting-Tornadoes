"""
Run this file from the "modelB" folder
"""

import uuid

import jax.numpy as jnp
import numpy as np

# NOTE-TO-SELF: think of polars as a fastter implementation of pandas
import polars as pl
import toml

from model import create_model_from_data


def main():
    # # load config
    # toml_file_path = "../data/params_A.toml"
    # with open(toml_file_path, "r") as toml_file:
    #     toml_params = toml.load(toml_file)
    
    print("Starting forecast.... Please stand by....")

    # get data
    clean_data_file_path = "../data/clean/cleaned_NOAA_SPC.csv"
    data = pl.read_csv(clean_data_file_path)
    # By default, all columns are inferred to be type "int64"

    # Construct model from the data
    # Only consider data from the month of April
    TARGET_MONTH = 4
    model = create_model_from_data(data, TARGET_MONTH)
    print(f"Intercept: {model.slope:.2f}, slope: {model.intercept:.2f}")
    
    # With normalized data, the target year is year 5
    TARGET_YEAR = 5
    mean_number_of_tornadoes_target_year = model.predict_value(TARGET_YEAR)
    print(f"Assuming a linear change over year, we predict {mean_number_of_tornadoes_target_year:.1f} tornadoes in the month of April")


if __name__ == "__main__":
    main()
