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
        "-n",
        "--host",
        dest="host",
        type=str,
        help="The hostname where the indexing service is hosted",
    )
    pushdata_parser.add_argument(
        "-p", "--port", dest="port", type=str, help="The port of the host"
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    # evaluate_parser.add_argument(
    #     "-c",
    #     "--configs",
    #     dest="configurations",
    #     type=list,
    #     help="List out the configurations that are required to evaluate",
    #     default=None,
    # )
    evaluate_parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        help="Be more verbose",
        action="store_true",
    )

    evaluate_parser = subparsers.add_parser("answer")
    evaluate_parser.add_argument(
        "-p",
        "--prompt",
        dest="prmopt",
        type=list,
        help="Write a prompt that you want to be answered about the EUR-lex corpus",
        default=None,
    )
    evaluate_parser.add_argument(
        "-o",
        "--use-openai",
        dest="use_openai",
        help="If you want to use `gpt-3.5-turbo`; you should pass this flag, otherwise `meta-llama/Llama-2-7b-chat-hf` from HuggingFace will be",
        action="store_true",
    )
    evaluate_parser.add_argument(
        "-a",
        "--apikey-file",
        dest="apikey_file",
        type=str,
        help="""
        Pass the API key by storing it in a file and pass the location of the file with this option,
        the API key of OpenAI, if you have selecd `gpt-3.5-turbo` with `--use-openai`, otherwise of API key of `HuggingFace`
        """,
    )

    try:
        args = parser.parse_args()

        match args.command:
            case "fetchdata":
                from llmlegalassistant.data import ArticlesScraper

                articles_scraper = ArticlesScraper(True)
                articles_scraper.fetch(args.export_type, args.no_samples)
            case "evaluate":
                if args.verbose:
                    print("Evaluation Starting...")

                from llmlegalassistant import LLMLegalAssistant

                if args.verbose:
                    print("Evaluation Started!")

                llmlegalassistant = LLMLegalAssistant(args.verbose)
                llmlegalassistant.evaluate()

                if args.verbose:
                    print("Evaluation Finished!")
            case "answer":
                from llmlegalassistant import LLMLegalAssistant

                llmlegalassistant = LLMLegalAssistant()
                llmlegalassistant.answer(
                    prompt=args.prompt,
                    is_openai=args.use_openai,
                    api_key_file=args.apikey_file,
                )
            case _:
                raise OSError(f"Unknown Command: {args.command}")
    except OSError:
        parser.print_help(sys.stderr)
        sys.exit(1)
