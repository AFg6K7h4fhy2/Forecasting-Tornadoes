"""
Linear Regression Model w/ Scipy.
"""

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
