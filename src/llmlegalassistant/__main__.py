import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLMLegalAsissatant - Assistant to solve legal queries"
    )
    subparsers = parser.add_subparsers(dest="command")

    fetchdata_parser = subparsers.add_parser("fetchdata")
    fetchdata_parser.add_argument(
        "-t",
        "--export-type",
        dest="export_type",
        type=str,
        help="Type of the result file of data fetched.",
    )
    fetchdata_parser.add_argument(
        "-n",
        "--no-samples",
        dest="no_samples",
        type=int,
        help="Number of samples to be created.",
        default=0,
    )

    pushdata_parser = subparsers.add_parser("pushdata")
    pushdata_parser.add_argument(
        "-i",
        "--index-name",
        dest="index_name",
        type=str,
        help="The name of the index to be created",
        default="index_articles",
    )
    pushdata_parser.add_argument(
        "-h",
        "--host",
        dest="host",
        type=str,
        help="The hostname where the indexing service is hosted",
    )
    pushdata_parser.add_argument(
        "-p", "--port", dest="port", type=str, help="The port of the host"
    )

    try:
        args = parser.parse_args()

        match args.command:
            case "fetchdata":
                from llmlegalassistant.data import ArticlesScraper

                articles_scraper = ArticlesScraper(True)
                articles_scraper.fetch(args.export_type, args.no_samples)
            case "pushdata":
                from llmlegalassistant.data import ArticlesIndexer

                articles_indexer = ArticlesIndexer(True, args.host, args.port)
                articles_indexer.create_index()
                articles_indexer.index(args.index_name)
            case _:
                raise OSError(f"Unknown Command: {args.command}")
    except OSError:
        parser.print_help(sys.stderr)
        sys.exit(1)
