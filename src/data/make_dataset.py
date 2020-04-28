# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import dask.bag as db
import dask
import dask.array as da
import zarr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path
import librosa
from typing import List, Tuple
import toolz as tz


def make_diagnosis_df(data_folder: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(
        data_folder.joinpath("patient_diagnosis.csv"),
        header=None,
        names=["Patient", "Diagnosis"],
    ).set_index("Patient")


def make_file_df(
    diagnosis_df: pd.DataFrame, audio_txt_folder: pathlib.Path
) -> pd.DataFrame:
    file_stats = list(
        map(
            lambda x: get_record_stats(audio_txt_folder, x.name.split(".")[0]),
            audio_txt_folder.glob("*.wav"),
        )
    )

    file_df = pd.DataFrame(
        file_stats,
        columns=[
            "Patient",
            "Section",
            "Location",
            "n_channels",
            "device",
            "filesize",
            "n_breaths",
        ],
    ).assign(
        Diagnosis=lambda x: x["Patient"]
        .astype(int)
        .map(lambda y: diagnosis_df.loc[y, "Diagnosis"])
    )
    return file_df


def get_record_stats(folder: str, file: str) -> Tuple[str, str, str, int, int]:
    name_elems = file.split("_")
    wav_size = Path(folder).joinpath(f"{file}.wav").stat().st_size
    # Count the lines, then subtract 1 cuz they end on an empty
    n_breath_cycles = (
        sum(1 for line in open(Path(folder).joinpath(f"{file}.txt"), "r")) - 1
    )
    return tuple(name_elems + [wav_size] + [n_breath_cycles])


def pad_to_length(arr: np.array, max_len: int) -> np.array:
    arr_len = arr.shape[0]
    diff = max_len - arr_len
    return np.pad(arr, (0, diff), mode="wrap")


def make_breath_array(
    audio_txt_folder: pathlib.Path, file_df: pd.DataFrame
) -> dask.array.Array:
    files_to_use = list(audio_txt_folder.rglob("*.wav"))
    # I downsampled it to the lowest I could get it without
    # running into DivideByZero errors.  Breathing is
    # low-frequency
    wav_bag = (
        db.from_sequence(files_to_use, npartitions=8)
        .map(lambda x: librosa.core.load(x, sr=87)[0])
        .compute()
    )

    max_len = max(x.shape[0] for x in wav_bag)

    breath_array = (
        db.from_sequence(wav_bag, npartitions=8)
        .map(lambda x: pad_to_length(x, max_len))
        .to_dataframe()
        .to_dask_array(lengths=True)
    )

    new_cols = da.stack(
        [
            da.from_array(file_df["n_breaths"].values),
            da.from_array((file_df["Diagnosis"] == "Healthy").astype(np.int8).values),
        ],
        axis=1,
    )

    return da.concatenate([breath_array, new_cols], axis=1).astype(np.float32)


def save_to_zarr(arr: dask.array.Array, folder: pathlib.Path, filename: str) -> None:
    destination = str(Path(folder, filename))
    da.to_zarr(arr.rechunk(), destination)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    raw_respiratory = Path(
        input_filepath,
        "raw",
        "respiratory-sound-database",
        "Respiratory_Sound_Database",
        "Respiratory_Sound_Database",
    )

    audio_txt_folder = raw_respiratory.joinpath("audio_and_txt_files")

    diagnosis_df = make_diagnosis_df(raw_respiratory)
    file_df = make_file_df(diagnosis_df, audio_txt_folder)
    breath_array = make_breath_array(audio_txt_folder, file_df)
    save_to_zarr(
        output_filepath, data_folder.joinpath("interim"), "breath_data_full.zarr"
    )

    train, test = train_test_split(
        breath_array.rechunk(),
        test_size=0.3,
        random_state=0,
        stratify=breath_array[:, -1],
    )

    save_to_zarr(train, output_filepath.joinpath("interim"), "breath_data_train.zarr")

    save_to_zarr(test, output_filepath.joinpath("interim"), "breath_data_test.zarr")

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()