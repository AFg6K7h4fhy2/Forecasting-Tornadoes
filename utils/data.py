"""
Produces a dataset of tornado counts
across the US by month, state, and year.
Requires files to be structured in a
particular manner.

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

import glob
import os

import numpy as np
import polars as pl

us_state_abbreviations = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "AND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


# associate numerical values with US states, starting from 1
us_state_values = dict(
    list(zip(us_state_abbreviations, list(range(1, len(us_state_abbreviations)+1))))
)

month_values = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def get_NOAA_SPC_data(
    data_read_path: str, 
    data_save_path: str, 
    use_preliminary: bool = True
) -> pl.DataFrame:
    """
    Converts raw NOAA Storm Prediction Center (SPC) into csv via polars.

    Parameters
    ----------
    data_read_path : str
        The path where raw NOAA SPC YYYY_Month_PorF files are saved.
    data_save_path : str
        The path to write the cleaned NOAA SPC pl.DataFrame csv to.
    use_preliminary : bool, optional
        Whether to use preliminary (instead of final) data. Defaults to True.

    Returns
    -------
    pl.DataFrame
        A dataframe with data from all raw NOAA SPC files
    """

    # collect data files in read path
    data_files = glob.glob(f"{data_read_path}*.txt")

    # read in files based on correct "finality", P = preliminary, F = final
    try:
        if use_preliminary:
            data_files = list(
                filter(
                    lambda path: path.split("/")[-1]
                    .split("_")[-1]
                    .replace(".txt", "")
                    == "P",
                    data_files,
                )
            )
        else:
            data_files = list(
                filter(
                    lambda path: path.split("/")[-1]
                    .split("_")[-1]
                    .replace(".txt", "")
                    == "F",
                    data_files,
                )
            )
    except Exception:
        print(
            "Either the data path is incorrect, or the files are not YYYY_Month_PorF"
        )

    # convert each file to pl.DataFrame
    successful_reads = 0
    dfs = []
    for dfile in data_files:
        with open(dfile, "r") as d:
            try:
                year, month, finality = dfile.split("/")[-1].split("_")
                lines = list(
                    filter(lambda line: line != "", d.read().split("\n"))
                )
                lines = [line.split("\t") for line in lines]
                header = lines[0]
                data = [
                    [
                        (
                            int(elt)
                            if elt not in us_state_abbreviations
                            else us_state_values[elt]
                        )
                        for elt in line
                    ]
                    for line in lines[1:]
                ]
                df = pl.DataFrame(dict(zip(header, np.asarray(data).T)))
                df = df.with_columns(
                    Year=pl.lit(int(year) - 2019),
                    Month=pl.lit(month_values[month]),
                )
                print(df)
                if isinstance(df, pl.DataFrame):
                    dfs.append(df)
                    successful_reads += 1
            except Exception as e:
                print(e)
        d.close()

    # if all files are successfully read, merge to a new df, then save to folder
    if len(data_files) == successful_reads:
        df_vertical_concat = pl.concat(dfs, how="vertical")
        if not os.path.exists(data_save_path):
            df_vertical_concat.write_csv(data_save_path)


# NOTE: should not be run if cleaned_NOAA_SPC.csv exists in data/clean
# get_NOAA_SPC_data("../data/raw/", "../data/clean/cleaned_NOAA_SPC.csv")