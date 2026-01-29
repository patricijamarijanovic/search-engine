"""Search endpoints."""

from fastapi import APIRouter

from sapien.core.model import Document
from sapien.core.neural_reranker import NeuralReranker
from sapien.core.rag_agent_gemini import RagAgentGemini
from sapien.core.search_engine import SearchEngine
from sapien.entrypoints.api.model import SearchResponse

router = APIRouter(tags=["search engine"])
_reranker = NeuralReranker()
_searcher = SearchEngine()
_rag_agent = RagAgentGemini()


@router.get("/search")
def search(query: str, num_results: int = 10) -> SearchResponse:
    """Search for documents matching the given query."""
    ranked = _searcher.search(query, top_k=100)
    """for document in ranked:
        print(f"DOC_ID {document['doc_id']}")
        print(f"SCORE: {document['score']}")
        print(f"TITLE: {document['title']}")
        print(f"TEXT: {document['text']}")
        print("-" * 60)"""

    reranked = _reranker.rerank(query, ranked, num_results)

    results: list[Document] = []
    for document in reranked:
        best_snippet: str = _reranker.get_best_snippet(query, document["text"])

        doc_id: int = document["doc_id"]
        title: str = document["title"]
        content: str =  document["text"]

        results.append(Document(id=doc_id, title=title, content=content, best_snippet=best_snippet))
        
    answer = None
    improved_query = None

    try:
        # Pokušavamo dobiti odgovor od AI-a
        is_question = _rag_agent.check_if_its_question(query).lower()
        
        if "yes" in is_question:  # Ponekad vrati "Yes." pa je bolje provjeriti sadrži li "yes"
            # Koristimo prvi (najrelevantniji) dokument za odgovor
            if results:
                answer = _rag_agent.answer_question_with_document(query, results[0].content)
                print(f"Answer to question: {answer}")
        
        improved_query = _rag_agent.improve_query_if_needed(query)

    except Exception as e:
        # Ako Google API (Gemini) vrati grešku (npr. 503 Overloaded), samo ispiši u konzolu
        # Aplikacija nastavlja dalje i vraća rezultate pretrage bez AI odgovora
        print(f"⚠️ AI Service Error (ignorirano): {e}")

    # --- KRAJ PROMJENE ---

    return SearchResponse(results=results, answer=answer, improved_query=improved_query)

    # if _rag_agent.check_if_its_question(query).lower() == "yes":
    #     answer = _rag_agent.answer_question_with_document(query, results[0].content)
    #     print(f"Answer to question: {answer}")
    # else:
    #     answer = None
    
    # improved_query: str | None = _rag_agent.improve_query_if_needed(query)

    # return SearchResponse(results=results, answer=answer, improved_query=improved_query)


@router.get("/search_like")
def search_like(doc_id: int, num_results: int = 10) -> SearchResponse:
    """Search for documents similar to the given document ID."""
    print(f"--->>> I'm searching for similar documents to {doc_id}....")
    similar_docs = _searcher.search_similar(doc_id, num_results)
    print(f"I searched for similar documents to {doc_id}....")
    print("SIMILAR DOCUMENTS: ")
    """for document in similar_docs:
        print(f"DOC_ID {document['doc_id']}")
        print(f"SCORE: {document['score']}")
        print(f"TITLE: {document['title']}")
        print(f"TEXT: {document['text']}")
        print("-" * 60)"""

    results: list[Document] = []
    for document in similar_docs:
        doc_id: int = document["doc_id"]
        title: str = document["title"]
        content: str = document["text"]
        best_snippet: str | None = None

        results.append(Document(id=doc_id, title=title, content=content, best_snippet=best_snippet))

    return SearchResponse(results=results)
