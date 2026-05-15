from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os
import shutil

CHROMA_PATH = "chroma"
CLEAN_DATA_PATH = "clean_data"


def main():

    documents = load_documents()

    chunks = split_text(documents)

    save_to_chroma(chunks)


def load_documents():

    loader = DirectoryLoader(
        CLEAN_DATA_PATH,
        glob="**/*.txt"
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    return documents


def split_text(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks")

    return chunks


def save_to_chroma(chunks):

    # Delete old database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Local embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create Chroma DB
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )

    db.persist()

    print("Saved to Chroma database!")


if __name__ == "__main__":
    main()