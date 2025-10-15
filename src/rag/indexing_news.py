import argparse
import os
import shutil
from datetime import datetime, date, time

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import TokenTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from src.rag.embedding import get_embedding_function
from src.rag.utils import generate_date_range
from src.config import DATA_PATH, START_DATE, END_DATE, PROVIDER, MODEL_NAME


CHROMA_PATH = 'chroma_new' #TODO temp
TEST_DATA_START_DATE = datetime(2024, 12, 18)

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


    filename = 'data/inc_news.csv' # TODO : use config
    documents = load_news_doc(filename)
    print(f'{len(documents)} news loaded from {filename}')
    inc_chunks = split_documents(documents)
    # print(inc_chunks[:3])
    print(f'{len(inc_chunks)} chunks to insert')
    add_to_chroma_batch(db, inc_chunks, True)

    filename = 'data/dec_news.csv' # TODO : use config
    documents = load_news_doc(filename)
    print(f'{len(documents)} news loaded from {filename}')
    dec_chunks = split_documents(documents)
    # print(dec_chunks[:3])
    print(f'{len(dec_chunks)} chunks to insert')
    add_to_chroma_batch(db, dec_chunks, False)
    
    print('✨ Finished indexing documents')


def load_news_doc(filename):
    df = pd.read_csv(filename)
    dfp = pd.read_csv(filename, parse_dates=['publish_at']) # TODO temporary solution
    df = df[dfp['publish_at'] < TEST_DATA_START_DATE]

    docs = DataFrameLoader(df, page_content_column="content")
    related_docs = [i for i in docs.load() if i.metadata['related_stocks'] != '[]']
    return related_docs


def add_to_chroma_batch(db, chunks: list[Document], is_price_increase):
    for i in range(0, len(chunks), 5000):
        batch = chunks[i:i+5000]
        print(f'Processing batch {i} to {i+len(batch)}')
        add_to_chroma(db, batch, is_price_increase)


def split_documents(documents: list[Document]):
    text_splitter = TokenTextSplitter(
        chunk_size=200, # 1 chinese char ~= 2~3 token?
        chunk_overlap=50,
        length_function=len,
        # is_separator_regex=False,
    )
    return extend_chunk_id(text_splitter.split_documents(documents))


def extend_chunk_id(chunks: Document):
    new_chunks = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["uuid"] = chunk.metadata["uuid"] + f':{i}'
        new_chunks.append(chunk)
    return new_chunks


def add_to_chroma(db, chunks: list[Document], is_price_increase):
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
            chunk.metadata = get_metadata(chunk, is_price_increase)
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def get_metadata(chunk, is_price_increase):
    return {"publish_at": datetime.strptime(chunk.metadata["publish_at"], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),
            "is_price_increase": is_price_increase}

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()