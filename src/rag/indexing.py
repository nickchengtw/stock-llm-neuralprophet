import argparse
import os
import shutil
from datetime import datetime, date

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import TokenTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from src.rag.embedding import get_embedding_function
from src.rag.utils import generate_date_range


CHROMA_PATH = "chroma"
DATA_PATH = "data"

# def main():
#     # Check if the database should be cleared (using the --clear flag).
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
#     if args.reset:
#         print("✨ Clearing Database")
#         clear_database()

#     start_date = date(2025, 3, 18)
#     end_date = date(2025, 6, 30)
#     for news_date in generate_date_range(start_date, end_date):
#         filename = f'data/news/news_{news_date}.csv' # TODO : use config
#         if not os.path.exists(filename):
#             print(f'No news data found for {news_date}')
#             continue
#         df = pd.read_csv(filename)
#         print(f'{len(df)} news loaded from {filename}')
#         documents = DataFrameLoader(df, page_content_column="content")
#         chunks = split_documents(documents.load())
#         print(f'{len(chunks)} chunks to insert')
#         add_to_chroma(chunks)

def main():
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()


    filename = 'rules.csv' # TODO : use config
    if not os.path.exists(filename):
        print(f'No news data found for')
    df = pd.read_csv(filename)
    print(f'{len(df)} news loaded from {filename}')
    documents = DataFrameLoader(df, page_content_column="content")
    chunks = split_documents(documents.load())
    print(f'{len(chunks)} chunks to insert')
    add_to_chroma(db, chunks)
    print('✨ Finished indexing documents')


def split_documents(documents: list[Document]):
    text_splitter = TokenTextSplitter(
        chunk_size=8192,
        chunk_overlap=80,
        # length_function=len,
        # is_separator_regex=False,
    )
    return extend_chunk_id(text_splitter.split_documents(documents))

def extend_chunk_id(chunks: Document):
    new_chunks = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["uuid"] = chunk.metadata["uuid"] + f':{i}'
        new_chunks.append(chunk)
    return new_chunks

def add_to_chroma(db, chunks: list[Document]):
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata['uuid'] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata['uuid'] for chunk in new_chunks]
        for chunk in new_chunks:
            chunk.metadata = get_metadata(chunk)
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def get_metadata(chunk):
    return {"publish_at": datetime.strptime(chunk.metadata["publish_at"], '%Y-%m-%d').strftime('%Y-%m-%d')}

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()