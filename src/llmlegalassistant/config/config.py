import os
from pathlib import Path


class Config:
    PROJECT_DIR = str(Path(__file__).resolve().parents[3])  # don't change

    # Change this to your dataset directory path
    DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")  # default, can change

    # Change this to your configs directory path
    CONFIG_DIR = os.path.join(PROJECT_DIR, "configs")  # default, can change

    # Evaluation dir
    EVALUATION_DIR = os.path.join(DATASET_DIR, "evaluation")  # default, can change

    METADATA_URI = os.path.join(
        DATASET_DIR, "metadata"
    )  # don't change, contains celex to fetch articles

    CELEX_URI = os.path.join(METADATA_URI, "celex.csv")

    # ARTICLES_URI = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"  # don't change

    # ARTICLES_PDF_URI = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:" # don't change

    ARTICLES_URI = "https://eur-lex.europa.eu/legal-content/EN/TXT/{TYPE}/?uri=CELEX:"  # don't change

    ARTICLES_DIR = os.path.join(DATASET_DIR, "articles")  # don't change

    REPLICATE_API_KEY = os.path.join(PROJECT_DIR, "replicate-api-key.txt")

    HF_API_KEY = os.path.join(PROJECT_DIR, "hf-api-key.txt")

    def get_article_file_dir(self, articles_dir: str, file_type: str) -> str:
        return os.path.join(articles_dir, file_type)
