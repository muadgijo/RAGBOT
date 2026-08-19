import argparse

from langchain_chroma import Chroma
from rag_utils import get_embedding_function

CHROMA_PATH = "chroma"


def main():

    # Read question from terminal
    parser = argparse.ArgumentParser()

    parser.add_argument("query_text", type=str)

    args = parser.parse_args()

    query_text = args.query_text

    # Load embedding model
    embedding_function = get_embedding_function()


    # Load Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search vector database
    results = db.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )

    print("\nTOP MATCHES:\n")

    for doc, score in results:

        print(f"Score: {score}")

        print(doc.page_content)

        print("\n-----------------\n")


if __name__ == "__main__":
    main()