""" 
File in need of overhaul. 
Pending improvement.
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
    "ND",
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

us_state_values = dict(
    list(zip(us_state_abbreviations, list(range(len(us_state_abbreviations))))))

month_values = {
    "January": 0,
    "February": 1,
    "March": 2,
    "April": 3,
    "May": 4,
    "June": 5,
    "July": 6,
    "August": 7,
    "September": 8,
    "October": 9,
    "November": 10,
    "December": 11,
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
                print(lines[1:])
                data = [
                    [
                        int(elt) if elt not in us_state_abbreviations else us_state_values[elt] for elt in line
                    ]
                    for line in lines[1:]
                ]
                df = pl.DataFrame(dict(zip(header, np.asarray(data).T)))
                df = df.with_columns(
                    Year=pl.lit(int(year)-2019), Month=pl.lit(month_values[month])
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
    
    


#get_NOAA_SPC_data("../data/raw/", "../data/clean/cleaned_NOAA_SPC.csv")