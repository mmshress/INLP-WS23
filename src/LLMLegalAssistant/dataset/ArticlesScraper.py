import requests
import pandas as pd

from bs4 import BeautifulSoup


class ArticlesScraper:
    def __init__():
        self.CSV_LOCATION = "data/metadata/CELEX-Category.csv"
        self.article_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"

    def scrape():
        metadata_df = self.create_metadata_dataframe()
        celex_column = self.get_column_from_dataframe(metadata_df, "CELEX_NUMBER")

        if len(celex_column) <= 0:
            return;

        for celex in celex_column:
            html_page = self.fetch_html_page_from_link(celex)
            formatted_html_page = self.format_html_page(html_page)
            self.save_file_location(celex, "html", "w", formatted_html_page)

    def save_file_to_location(file_name: str, file_type: str, mode: str, file_content: str) -> bool:
        try:
            with open(''.join([self.CSV_LOCATION, '.'.join([file_name, file_type])]), mode) as file:
                file.write(file_content)
            return True
        except as error:
            return False

    def format_html_page(html: str) -> str:
        return BeautifulSoup(html, "html.parser").get_text()

    def fetch_html_page_from_link(id: str) -> str | None:
        response = requests.get(''.join([self.article_url, id]))
        if response is not None:
            return response.text
        return None

    def get_column_from_dataframe(df: pd.Dataframe, column: str) -> pd.Dataframe:
        if column in df.columns:
            return df[f"{column}"]
        return pd.Dataframe()

    def create_metadata_dataframe():
        metadata_df = pd.read_csv(self.CSV_LOCATION)
        metadata_df.columns = ["index", "CELEX_NUMBER", "category"]
        return metadata_df

