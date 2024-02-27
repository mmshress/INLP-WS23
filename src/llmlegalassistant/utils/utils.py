import os
import shutil

import pandas as pd
import yaml

from llmlegalassistant.config import Config

config = Config()


PROJECT_DIR = config.PROJECT_DIR
CELEX_URI = config.CELEX_URI
ARTICLES_DIR = config.ARTICLES_DIR
DATASET_DIR = config.DATASET_DIR
ARTICLES_URI = config.ARTICLES_URI
CONFIG_DIR = config.CONFIG_DIR
ARTICLE_FILE_DIR = ARTICLES_DIR


def get_project_dir() -> str:
    return PROJECT_DIR


def get_articles_uri(type: str = "HTML") -> str:
    return ARTICLES_URI.format(TYPE=type)


def get_dataset_dir() -> str:
    return DATASET_DIR


def get_articles_dir() -> str:
    return ARTICLES_DIR


def get_file_dir(file_type: str) -> str:
    # get article directory based on the file type
    ARTICLE_FILE_DIR = config.get_article_file_dir(
        ARTICLES_DIR, "".join([file_type, "s"])
    )

    return ARTICLE_FILE_DIR


def load_configurations(configurations: list[str] | None = None) -> list[dict]:
    """
    This function returns the corresponding configuration from
    the config file at the project root the configuration file
    is used for evaluation

    Parameters
    ----------
    configurations : list[str]
        list of configurations that are needed if none is passed
        then all the configurations are expected

    Returns
    -------
    list[dict]
        returns all configurations or only the ones that are requested
    """
    configs = []

    config_file = os.path.join(CONFIG_DIR, "configs.yaml")
    with open(config_file, "r") as file:
        configs = yaml.safe_load(file)

    if configurations is not None:
        return [config for config in configs if config["config"] in configurations]

    return configs


def get_files(file_type: str) -> list[str]:
    # get article directory based on the file type
    ARTICLE_FILE_DIR = config.get_article_file_dir(
        ARTICLES_DIR, "".join([file_type, "s"])
    )

    return os.listdir(ARTICLE_FILE_DIR)


def create_and_get_empty_file_dir(file_type: str) -> str:
    # get article directory based on the file type
    ARTICLE_FILE_DIR = config.get_article_file_dir(
        ARTICLES_DIR, "".join([file_type, "s"])
    )

    try:
        # makes article file type dir if doesn't exists
        os.makedirs(ARTICLE_FILE_DIR, exist_ok=False)
    except OSError:
        # If article file type dir already exists,
        # removes all old articles
        shutil.rmtree(ARTICLE_FILE_DIR)
        os.makedirs(ARTICLE_FILE_DIR, exist_ok=True)

    return ARTICLE_FILE_DIR


def get_metadata() -> pd.DataFrame | None:
    metadata_df = pd.read_csv(CELEX_URI)
    if len(metadata_df):
        return metadata_df

    return None


def get_column(df: pd.DataFrame, column: str) -> pd.Series | pd.DataFrame | None:
    try:
        column_series = df[column]
        if not len(column_series):
            raise KeyError(f"Column {column} is empty!")
    except KeyError:
        return None

    return column_series


def get_api_key(service: str) -> str | None:
    api_key = ""
    match service:
        case "HuggingFace":
            with open(config.HF_API_KEY, "r") as file:
                api_key = file.read()
        case "Replicate":
            with open(config.REPLICATE_API_KEY, "r") as file:
                api_key = file.read()
        case _:
            return None

    return api_key
