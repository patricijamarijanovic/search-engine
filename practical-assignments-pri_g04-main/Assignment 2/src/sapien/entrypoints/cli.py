import argparse
import logging

from sapien.core.indexer import Indexer
from sapien.core.limit_memory import start_memory_monitor
from sapien.core.logging import setup_logging

setup_logging(logging.INFO)
start_memory_monitor(show_memory_updates=True)


"""
    Function that defines all arguments and sets their default values.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sapien Indexer CLI")

    # basic arguments for the indexer
    parser.add_argument("file_path", type=str, help="Path to the file to index")
    parser.add_argument(
        "--min_term_freq", type=int, default=5, help="Minimum term frequency in document to store"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./output",
        help="Directory to store generated files",
    )

    # tokenizer arguments
    parser.add_argument(
        "--separate_alphanumeric",
        dest="separate_alphanumeric",
        action="store_true",
        help="Separate alphanumeric tokens (default: True)",
    )
    parser.add_argument(
        "--no-separate_alphanumeric",
        dest="separate_alphanumeric",
        action="store_false",
        help="Do not separate alphanumeric tokens",
    )

    parser.add_argument(
        "--remove_numbers",
        dest="remove_numbers",
        action="store_true",
        help="Remove numeric-only tokens (default: True)",
    )
    parser.add_argument(
        "--no-remove_numbers",
        dest="remove_numbers",
        action="store_false",
        help="Keep numeric-only tokens",
    )

    parser.add_argument(
        "--remove_URLs", dest="remove_URLs", action="store_true", help="Remove URLs (default: True)"
    )
    parser.add_argument(
        "--no-remove_URLs", dest="remove_URLs", action="store_false", help="Keep URLs"
    )

    parser.add_argument(
        "--remove_emails",
        dest="remove_emails",
        action="store_true",
        help="Remove emails (default: True)",
    )
    parser.add_argument(
        "--no-remove_emails", dest="remove_emails", action="store_false", help="Keep emails"
    )

    parser.add_argument(
        "--lowercase",
        dest="lowercase",
        action="store_true",
        help="Convert to lowercase (default: True)",
    )
    parser.add_argument(
        "--no-lowercase", dest="lowercase", action="store_false", help="Keep case as-is"
    )

    parser.add_argument(
        "--stemmer", dest="stemmer", action="store_true", help="Enable stemming (default: True)"
    )
    parser.add_argument(
        "--no-stemmer", dest="stemmer", action="store_false", help="Disable stemming"
    )

    parser.add_argument(
        "--stopwords",
        dest="stopwords",
        action="store_true",
        help="Remove stopwords (default: True)",
    )
    parser.add_argument(
        "--no-stopwords", dest="stopwords", action="store_false", help="Keep stopwords"
    )

    parser.set_defaults(
        separate_alphanumeric=True,
        remove_numbers=True,
        remove_URLs=True,
        remove_emails=True,
        lowercase=True,
        stemmer=True,
        stopwords=False,
    )

    return vars(parser.parse_args())


def main():
    arguments = parse_arguments()
    indexer = Indexer(**arguments)
    indexer.store_metadata()

    indexer.create_inverted_index()
    indexer.create_forward_index()


if __name__ == "__main__":
    main()
