import os
import requests

import pandas as pd

from bs4 import BeautifulSoup
from markdownify import markdownify as md


class ArticlesScraper:
    def __init__(self) -> None:
        self.CSV_LOCATION = "data/metadata/CELEX-Category.csv"
        self.ARTICLE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"

    def fetch(self, output_dir: str, output_file_type: str, no_samples: int) -> None:
        metadata_df = self.get_metadata()
        celex_column = self.get_column(
            metadata_df.head(no_samples) if no_samples != 0 else metadata_df, 
            "celexnumber"
        )

        if len(celex_column) <= 0:
            return

        for celex_number in celex_column:
            origin_article = self.fetch_article(celex_number)
            if origin_article is None:
                continue
            article_content = self.parse_content(origin_article, output_file_type)
            article_location = self.generate_article_location(output_dir, celex_number, output_file_type)
            self.save_article(article_location, article_content)

    def get_metadata(self) -> pd.DataFrame:
        metadata_df = pd.read_csv(self.CSV_LOCATION)
        metadata_df.columns = ["index", "celexnumber", "category"]
        return metadata_df 

    def get_column(self, df: pd.Dataframe, column: str) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series()

    def fetch_article(self, celex_number: str) -> str | None:
        response = requests.get("".join([self.ARTICLE_URL, celex_number]))
        if response is not None:
            return response
        return None

    def generate_article_location(self, output_dir: str, celex: str, output_file_type: str) -> str:
        return os.path.join(output_dir, ".".join([celex, output_file_type]))

    def save_article(self, file_name: str, article_content: str, mode: str = "w") -> bool:
        try:
            with open(file_name, mode) as article:
                article.write(article_content)
            return True
        except OSError:
            return False

    def parse_content(self, article_content: str, export_type: str) -> str:
        if export_type == "md":
            return md(article_content)
        elif export_type == "html":
            return BeautifulSoup(article_content, "html.parser").prettify()
        else:
            return BeautifulSoup(article_content, "html.parser").get_text()
