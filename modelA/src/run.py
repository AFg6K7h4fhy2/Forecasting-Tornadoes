"""
File quite messy.
In need of major overhaul. 
"""


import os
import uuid
import polars as pl
import toml
from data import get_NOAA_SPC_data
from modelA import inference, tornado_modelA, posterior_predictive_distribution
import jax.numpy as jnp
# from vis import vis_tornado_data, vis_prediction


def main():

    # load config
    toml_file_path = "../data/params.toml"
    with open(toml_file_path, "r") as toml_file:
        cf = toml.load(toml_file)
        toml_file.close()

    # get data
    clean_data_file_path = "../data/clean/cleaned_NOAA_SPC.csv"
    raw_data_path = "../data/raw/"
    if not os.path.exists(clean_data_file_path):
        # get_NOAA_SPC_data(
        #     raw_data_path, clean_data_file_path, cf["data"]["preliminary"]
        # )
        pass
    data = pl.read_csv(clean_data_file_path)


    # run inference
    uuid4 = uuid.uuid4()
    save_path = f"../output/samples/{uuid4}.csv"
    samples = inference(tornado_modelA, cf, data, save_path)
    postP = posterior_predictive_distribution(
        samples, tornado_modelA, cf)
    x = {k: jnp.percentile(v, 2.5, axis=0) for k, v in postP.items()}
    print(postP["obs"].shape)
    print(jnp.mean(postP["obs"], axis=0))
    print(jnp.sum(postP["obs"], axis=0))
    print(np.mean(jnp.sum(postP["obs"], axis=0)))
    # print(jnp.sum(jnp.mean(postP["obs"], axis=0), axis=1))
    # print(jnp.percentile(jnp.sum(postP["obs"], axis=0), 2.5, axis=0))
    # print(postP)
    # data = {
    #             param: samples.__array__() for param, samples in data.items()
    #         }
    #         df = pl.DataFrame(data)
    #         return df
    #     except Exception as e:
    #         print(f"{e}")

    # def save_df_as_json(df, save_path):
    #     df.write_json(save_path)


if __name__ == "__main__":
    main()
