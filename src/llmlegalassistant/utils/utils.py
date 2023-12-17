import os
import shutil
import yaml
import pandas as pd

from llmlegalassistant.config import Config


class Utils:
    def __init__(self, verbose: bool):
        self.verbose = verbose

        self.config = Config()

        self.PROJECT_DIR = self.config.PROJECT_DIR
        self.CELEX_URI = self.config.CELEX_URI
        self.ARTICLES_DIR = self.config.ARTICLES_DIR
        self.DATASET_DIR = self.config.DATASET_DIR
        self.ARTICLES_URI = self.config.ARTICLES_URI
        self.ARTICLE_FILE_DIR = self.ARTICLES_DIR

        self.get_article_file_dir = self.config.get_article_file_dir

    def get_project_dir(self) -> str:
        return self.PROJECT_DIR

    def get_articles_uri(self) -> str:
        return self.ARTICLES_URI

    def get_dataset_dir(self) -> str:
        return self.DATASET_DIR

    def get_articles_dir(self) -> str:
        return self.ARTICLES_DIR

    def get_file_dir(self, file_type) -> str:
        self.ARTICLE_FILE_DIR = self.get_article_file_dir(self.ARTICLES_DIR, file_type+"s")

        try:
            # makes article file type dir if doesn't exists
            os.makedirs(self.ARTICLE_FILE_DIR, exist_ok=False)
        except OSError:
            # If article file type dir already exists, 
            # removes all old articles
            article_path = os.path.join(self.ARTICLE_FILE_DIR, "{file}")
            if self.verbose:
                print("[Utils] File directory already exists!")

            if self.verbose:
                print("[Utils] Removing old articles...")

            for file in os.listdir(self.ARTICLE_FILE_DIR):

                file_path = article_path.format(file)
                if os.isfile(file_path):
                    os.remove(file_path)
                elif os.isdir(file_path):
                    shutil.rmtree(file_path)

            if self.verbose:
                print("[Utils] Removed old articles!")
        finally:
            return self.ARTICLE_FILE_DIR

    def get_metadata(self) -> pd.DataFrame | None:
        if self.verbose:
            print("[Utils] Loading metadata...")

        metadata_df = pd.read_csv(self.CELEX_URI)
        if len(metadata_df):
            if self.verbose:
                print("[Utils] Metadata loaded!")
            return metadata_df

        if self.verbose:
            print("[Utils] Metadata not found!")

        return None

    def get_column(self, df: pd.DataFrame, column: str) -> pd.Series | None:
        if self.verbose:
            print(f"[Utils] Loading {column} column...")
        
        try:
            column_series = df[column]
            if not len(column_series):
                raise KeyError(f"Column {column} is empty!")

            if self.verbose:
                print(f"[Utils] Column {column} loaded!")
        except KeyError:
            if self.verbose:
                print(f"[Utils] Column {column} not found!")
            return None

        return column_series
