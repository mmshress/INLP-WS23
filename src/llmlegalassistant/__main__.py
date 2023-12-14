import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMLegalAsissatant - Assistant to solve legal queries")
    subparsers = parser.add_subparsers(dest="command")

    fetchdata_parser = subparsers.add_parser('fetch-data')
    fetchdata_parser.add_argument("-t", "--export-type", dest="export_type", type=str, help="Type of the result file of data fetched.")
    fetchdata_parser.add_argument("-d", "--output-dir", dest="output_dir", type=str, help="Directory of the data to be exported.")
    fetchdata_parser.add_argument("-n", "--no-samples", dest="no_samples", type=int, help="Number of samples to be created.")

    try:
        args = parser.parse_args()
    except OSError:
        parser.print_help(sys.stderr)
        sys.exit(2)

    match args.command:
        case "fetch-data":
            from llmlegalassistant.data import ArticlesScraper
            articles_scraper = ArticlesScraper()
            articles_scraper.fetch(args.output_dir, args.export_type, args.no_samples)
        case _:
            print(f"Unknown Command: {args.command}")
            parser.print_help(sys.stderr)
            sys.exit(1)
