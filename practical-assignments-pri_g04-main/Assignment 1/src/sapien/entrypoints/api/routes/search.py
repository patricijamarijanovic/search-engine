"""Search endpoints."""

from fastapi import APIRouter

from sapien.core.model import Document
from sapien.core.search_engine import SearchEngine
from sapien.entrypoints.api.model import SearchResponse

router = APIRouter(tags=["search engine"])


@router.get("/search")
def search(query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents matching the given query."""
    searcher = SearchEngine()
    ranked = searcher.search(query, top_k=num_results)
    '''for document in ranked:
        print(f"DOC_ID {document['doc_id']}")
        print(f"SCORE: {document['score']}")
        print(f"TITLE: {document['title']}")
        print(f"TEXT: {document['text']}")
        print("-" * 60)'''

    #print(f"Printing ranked: {ranked}")
    
    # testiranje bzvz radi li ako se odabere prvi dokument za trazenje slicnog njemu
    '''doc_id = ranked[0]["doc_id"]
    print(f"--->>> I'm searching for similar documents to {doc_id}....")
    similar_docs = searcher.search_similar(doc_id, 3)
    print(f"I searched for similar documents to {doc_id}....")
    for document in similar_docs:
        print(f"DOC_ID {document['doc_id']}")
        print(f"SCORE: {document['score']}")
        print(f"TITLE: {document['title']}")
        print(f"TEXT: {document['text']}")
        print("-" * 60)'''
    
    results: list[Document] = []
    for document in ranked:
        doc_id: int = document['doc_id']
        title: str = document['title']
        content: str = document['text']

        results.append(Document(id=doc_id, title=title, content=content))

    return SearchResponse(results=results)


@router.get("/search_like")
def search_like(doc_id: int, num_results: int = 10) -> SearchResponse:
    """Search for documents similar to the given document ID."""
    searcher = SearchEngine()
    
    print(f"--->>> I'm searching for similar documents to {doc_id}....")
    similar_docs = searcher.search_similar(doc_id, num_results)
    print(f"I searched for similar documents to {doc_id}....")
    print("SIMILAR DOCUMENTS: ")
    '''for document in similar_docs:
        print(f"DOC_ID {document['doc_id']}")
        print(f"SCORE: {document['score']}")
        print(f"TITLE: {document['title']}")
        print(f"TEXT: {document['text']}")
        print("-" * 60)'''

    results: list[Document] = []
    for document in similar_docs:
        doc_id: int = document['doc_id']
        title: str = document['title']
        content: str = document['text']

        results.append(Document(id=doc_id, title=title, content=content))

    return SearchResponse(results=results)
