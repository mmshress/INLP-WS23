import os

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

from llmlegalassistant.utils import (
    get_articles_uri,
    get_column,
    get_file_dir,
    get_metadata,
)


class ArticlesScraper:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

        # EUR-LEX doesn't return a 404 status code
        self.DOES_NOT_EXIST_STRING = "The requested document does not exist."

    def fetch(self, output_file_type: str, no_samples: int) -> None:
        # contains celex number and category of the article
        metadata_df = get_metadata()
        if metadata_df is None:
            return

        # get all or specified number of samples
        celex_column = get_column(
            metadata_df.head(no_samples) if no_samples != 0 else metadata_df,
            "celexnumber",
        )

        if celex_column is None or len(celex_column) <= 0:
            return

        if self.verbose:
            print("[ArticlesScraper] Downloading articles...")

        # Create output dir or remove all files inside it
        output_dir = get_file_dir(output_file_type)
        for celex_number in tqdm(celex_column, desc="Downloading", unit="article"):
            # Fetch the articles
            origin_article = self.fetch_article(celex_number, output_file_type.upper())

            if origin_article is None:
                continue

            article_content = self.parse_content(origin_article, output_file_type)
            if article_content is None:
                continue

            article_location = self.generate_article_location(
                output_dir, celex_number, output_file_type
            )

            self.save_article(article_location, article_content)

        if self.verbose:
            print(f"[ArticleScraper] {output_file_type} Articles Downloaded!")

    def fetch_article(
        self, celex_number: str, type: str = "HTML"
    ) -> requests.Response | None:
        response = requests.get("".join([get_articles_uri(type), celex_number]))
        if response is not None and self.DOES_NOT_EXIST_STRING not in response.text:
            return response

        if self.verbose:
            print(f"[ArticleScraper] Article {celex_number} doesn't exists!")

        return None

    def generate_article_location(
        self, output_dir: str, celex: str, output_file_type: str
    ) -> str:
        return os.path.join(output_dir, ".".join([celex, output_file_type]))

    def save_article(
        self, file_name: str, article_content: str | bytes, mode: str = "w"
    ) -> bool:
        try:
            with open(file_name, mode) as article:
                article.write(article_content)
        except (OSError, TypeError):
            return False

        return True

    def parse_content(
        self, article_content: requests.Response, export_type: str
    ) -> str | bytes | None:
        if export_type == "pdf":
            return article_content.content
        content = article_content.text
        if export_type == "md":
            return md(content)
        elif export_type == "html":
            return BeautifulSoup(content, "html.parser").prettify()
        elif export_type in ["text", "txt"]:
            # return article_content.text
            return BeautifulSoup(content, "html.parser").get_text()

        return None
