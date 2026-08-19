from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rag_utils import get_embedding_function


import os
import shutil

CHROMA_PATH = "chroma"
CLEAN_DATA_PATH = "clean_data"


def main():

    documents = load_documents()

    chunks = split_text(documents)

    save_to_chroma(chunks)


def load_documents():

    # Avoid unstructured-based loaders (they require extra NLTK models like punkt_tab)
    # so simple .txt files load reliably.
    # Use plain TextLoader to prevent DirectoryLoader from falling back to unstructured-based loaders
    # (which require extra NLTK assets like punkt_tab).
    loader = DirectoryLoader(
        CLEAN_DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=False,
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

    # Embedding model
    embedding_function = get_embedding_function()

    # Create Chroma DB
    Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH,
    )

    print("Saved to Chroma database!")



if __name__ == "__main__":
    main()