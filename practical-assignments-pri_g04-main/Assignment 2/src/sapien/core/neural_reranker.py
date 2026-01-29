# dodatak za patricijin komp, obrisi sebi poslije ak ce smetat
import os
from time import time

import torch
from sentence_transformers import CrossEncoder

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class NeuralReranker:
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        # print(f"Loading reranker model: {model_name}")
        # print(torch.__version__)
        # print(torch.version.cuda)
        # print(f"Cuda available: {torch.cuda.is_available()}")

        # dodatak za patricijin komp, obrisi sebi poslije ak ce smetat
        torch.set_num_threads(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Na Macu je često bolje ostati na CPU za male modele ako MPS radi probleme
        self.model = CrossEncoder(model_name, device=device)
        print(f"Reranker ready on {device} with single-thread mode!")

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = CrossEncoder(model_name, device=device)
        # print("Reranker ready!")

    def rerank(self, query: str, documents: list[dict], num_results: int = 10) -> list[dict]:
        """Re-ranks documents by semantic relevance to the query.
        Each document must have a 'text' field.
        Returns list sorted from most to least relevant.
        """

        print(f"Reranking {len(documents)} documents")
        print("CrossEncoder device:", self.model.model.device)

        # create pairs (query, text)
        pairs = [(query, doc["text"]) for doc in documents]

        start_time = time()
        # get scores
        scores = self.model.predict(pairs)
        end_time = time()
        print(f"Reranked in {end_time - start_time} seconds")
        # add score for each document
        for doc, score in zip(documents, scores):
            doc["neural_score"] = float(score)

        return sorted(documents, key=lambda x: x["neural_score"], reverse=True)[:num_results]

    def segment_text(self, text: str) -> list[str]:
        # Razdvaja tekst po dvostrukim prijelomima redaka, uklanjajući prazne sekcije
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def get_best_snippet(self, query: str, document_text: str) -> str:
        paragraphs = self.segment_text(document_text)

        if not paragraphs:
            return document_text[:200] + "..."  # Povratak kratkog isječka ako nema paragrafa

        # Stvaranje parova (upit, paragraf) za Cross-Encoder
        pairs = [(query, p) for p in paragraphs]

        # Dobivanje semantičkih rezultata
        scores = self.model.predict(pairs)

        # Pronalaženje indeksa odlomka s najvećim rezultatom
        best_score_index = scores.argmax()

        return paragraphs[best_score_index]  # Vraćanje najboljeg odlomka
