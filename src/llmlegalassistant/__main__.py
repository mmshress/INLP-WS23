import argparse
import sys

from llmlegalassistant.utils import load_configurations


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
    evaluate_parser.add_argument(
        "-c",
        "--configurations",
        dest="configurations",
        type=list,
        help="List out the configurations that are required to evaluate",
        default=None,
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
            case "evaluate":
                from llmlegalassistant import LLMLegalAssistant

                llmlegalassistant = LLMLegalAssistant(verboe=True)

                configurations = args.configurations
                configs = load_configurations(configurations)

                for config in configs:
                    embed_model = None

                    splitter = config["splitter"]["type"]
                    model_name = config["embed"]
                    index_name = config["store"]["index"]
                    retriever = config["retriever"]["type"]
                    # will need later when we add llm models
                    # llm_model = config["model"]["name"]
                    if model_name is not None:
                        from langchain.embeddings.huggingface import \
                            HuggingFaceEmbeddings

                        embed_model = HuggingFaceEmbeddings(model_name=model_name)

                    if splitter is not None:
                        from llmlegalassistant.splitter import SplitterFactory

                        chunk_size = config["splitter"]["chunk_size"]
                        overlap_size = config["splitter"]["overlap_size"]
                        textsplitter = SplitterFactory.generate_splitter(
                            splitter=splitter,
                            embed_model=embed_model,
                            chunk_size=chunk_size,
                            overlap_size=overlap_size,
                        )

                    if retriever is not None:
                        from llmlegalassistant.retriever import \
                            RetrieverFactory

                        top_k = config["retriever"]["top_k"]
                        num_queries = config["retriever"]["num_queries"]
                        retriever = RetrieverFactory.generate_retriver(
                            document_retriever=retriever,
                            top_k=top_k,
                            num_queries=num_queries,
                        )

                    llmlegalassistant.generate_query(
                        textsplitter, embed_model, index_name, retriever
                    )
            case _:
                raise OSError(f"Unknown Command: {args.command}")
    except OSError:
        parser.print_help(sys.stderr)
        sys.exit(1)
