from langchain_core.documents import Document

from src.rag.indexing import split_documents

def test_split_documents():
    # Test with short content
    short_doc = Document(
        page_content='短內容',
        metadata={'uuid': 'test1', 'publish_at': '2025-04-02 16:00:00'}
    )
    short_chunks = split_documents([short_doc])
    print("Short chunks:", short_chunks)
    assert len(short_chunks) == 1
    assert 'uuid' in short_chunks[0].metadata

    # Test with long content
    long_content = '長內容' * 3000
    long_doc = Document(
        page_content=long_content,
        metadata={'uuid': 'test2', 'publish_at': '2025-04-02 16:00:00'}
    )
    long_chunks = split_documents([long_doc])
    print("Long chunks:", long_chunks)
    assert len(long_chunks) > 1
    assert long_chunks[0].metadata['uuid'] == 'test2:0'
    assert long_chunks[1].metadata['uuid'] == 'test2:1'
