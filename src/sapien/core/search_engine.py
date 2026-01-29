import json
import math
import os
import sqlite3
from collections import defaultdict
from time import time

from sapien.core.tokenizer import Tokenizer



class SearchEngine:
    def __init__(
        self,
        index_path: str = "output/final_index.jsonl",
        documents_stats_path: str = "output/documents_stats.jsonl",
        documents_metadata_path: str = "output/documents_metadata.jsonl",
        indexer_metadata_path: str = "output/indexer_metadata.jsonl",
        offset_index_path: str = "output/offset_index.json",
    ):
        self.index_path = index_path
        self.documents_stats_path = documents_stats_path
        self.documents_metadata_path = documents_metadata_path
        self.indexer_metadata_path = indexer_metadata_path
        self.offset_index_path = offset_index_path
        self.index = defaultdict(list)
        self.documents_lengths = {}
        self.document_count = 0
        self.average_document_length = 0
        self.total_tokens = 0
        self.tokenizer_metadata = {}

        
        self.load_documents_stats()
        self.load_documents_metadata()
        self.load_indexer_metadata()
        self.load_offset_index()

        self.tokenizer = Tokenizer(**self.tokenizer_metadata)

        
    def load_index(self):
        print(f"Loading index from {self.index_path}...")
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                term, postings = next(iter(data.items()))
                self.index[term] = postings


    def load_documents_stats(self):
        start_time: float = time()
        with open(self.documents_stats_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.documents_lengths[data["doc_id"]] = data["length"]

        end_time: float = time()

        print(f"Loaded file with documents stats in {end_time - start_time} seconds.")


    def load_documents_metadata(self):
        start_time: float = time()
        with open(self.documents_metadata_path, encoding="utf-8") as f:
            data = json.load(f)
            self.document_count = data.get("doc_count", 0)
            self.average_document_length = data.get("avg_doc_length", 0)
            self.total_tokens = data.get("total_tokens", 0)

        end_time: float = time()
        print(f"Loaded file with documents metadata in {end_time - start_time} seconds.")


    def load_indexer_metadata(self):
        start_time: float = time()
        with open(self.indexer_metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

            tokenizer_metadata = {
                "separate_alphanumeric": metadata.get("separate_alphanumeric", False),
                "remove_numbers": metadata.get("remove_numbers", False),
                "remove_URLs": metadata.get("remove_URLs", False),
                "remove_emails": metadata.get("remove_emails", False),
                "min_token_length": metadata.get("min_token_length", 1),
                "lowercase": metadata.get("lowercase", False),
                "stemmer": metadata.get("stemmer", False),
                "use_stopwords": metadata.get("stopwords", False),
            }

        self.tokenizer_metadata = tokenizer_metadata
        end_time: float = time()
        print(f"Loaded file with indexer metadata in {end_time - start_time} seconds.")


    def load_offset_index(self):
        start_time: float = time()
        self.offsets = {}
        with open(self.offset_index_path, encoding="utf-8") as f:
            self.offsets = json.load(f)

        end_time: float = time()
        print(f"Loaded file with index offset in {end_time - start_time} seconds.")


    def get_term_postings(self, term: str):
        pos = self.offsets.get(term, None)

        if pos is None:
            print(f"No postings for term: {term}")
            return []

        # 2. Skoči na offset u final_index.jsonl i učitaj postings
        with open(self.index_path, "rb") as f:
            f.seek(pos)
            line = f.readline().decode("utf-8")
            data = json.loads(line)
            postings = next(iter(data.values()))
            return postings


    def check_database(self, database_path="output/forward_index.db"):
        if not os.path.exists(database_path):
            print(f"Database file not found at {database_path}")
            return False

        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Dohvati sve tablice u bazi
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # get row with given doc_id
        doc_id = 2
        cursor.execute(f"SELECT title, text FROM documents WHERE doc_id = {doc_id}")
        result = cursor.fetchone()
        print(result)

        conn.close()

        if not tables:
            print(f"Database at {database_path} is empty (no tables).")
            return False

        print(f"Database loaded successfully! Tables found: {[t[0] for t in tables]}")

        return True


    def get_document_by_id(doc_id: int, database_path="output/forward_index.db"):
        if not os.path.exists(database_path):
            print(f"Database file not found at {database_path}")
            return False

        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Dohvati sve tablice u bazi
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # get row with given doc_id
        cursor.execute(f"SELECT title, text FROM documents WHERE doc_id = {doc_id}")
        result = cursor.fetchone()
        # print(result)

        conn.close()

        if result:
            title, text = result
            return {"doc_id": doc_id, "title": title, "text": text}
        else:
            return None


    def get_documents_by_ids(doc_ids, database_path="output/forward_index.db"):
        if not os.path.exists(database_path):
            print(f"Database file not found at {database_path}")
            return []

        if not doc_ids:
            return []

        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        placeholders = ",".join(["?"] * len(doc_ids))
        query = f"SELECT doc_id, title, text FROM documents WHERE doc_id IN ({placeholders})"
        cursor.execute(query, doc_ids)

        results = cursor.fetchall()
        conn.close()

        documents = [
            {"doc_id": doc_id, "title": title, "text": text} for doc_id, title, text in results
        ]
        return documents


    def search_tokenized(self, tokens: list[str], top_k: int = 100, k: float = 1.2, b: float = 0.75):
        scores = {}  # bm25 score for each doc_id is stored here

        for token in tokens:
            postings = self.get_term_postings(token)
            if not postings:
                continue

            df = len(postings)  # in how many documents is the term
            idf = math.log(1 + (self.document_count - df + 0.5) / (df + 0.5))
            # ili math.log(self.document_count / df)
            # tako je u prezi, a svuda drugdje na internetu ovo nekomentirano ?

            for doc_id, tf in postings:
                doc_len = self.documents_lengths[doc_id]
                score = (
                    idf
                    * (tf * (k + 1))
                    / (tf + k * ((1 - b) + b * (doc_len / self.average_document_length)))
                )
                scores[doc_id] = scores.get(doc_id, 0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = ranked[:top_k]
        doc_ids = [doc_id for doc_id, _ in top_docs]

        documents = SearchEngine.get_documents_by_ids(doc_ids)

        doc_dict = {doc["doc_id"]: doc for doc in documents}
        ranked_documents = []
        for doc_id, score in top_docs:
            if doc_id in doc_dict:
                doc = doc_dict[doc_id]
                doc["score"] = score
                ranked_documents.append(doc)

        return ranked_documents


    def search(self, query: str, top_k: int = 100, k: float = 1.2, b: float = 0.75) -> list[dict]:
        print(f"this is query before tokenization: {query}")
        tokens = self.tokenizer.tokenize(query)
        print(f"tokens: {tokens}")
            
        return self.search_tokenized(tokens, top_k, k, b)


    # calculate tf-idf weight for each token in a document/query
    # to find most important tokens in a document
    def calculate_tf_idf(self, doc_id: int):
        document = SearchEngine.get_document_by_id(doc_id)
        doc_tokenized = set(self.tokenizer.tokenize(document["text"]))

        tf_idf_weights = {}
        for term in doc_tokenized:
            postings = self.get_term_postings(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log(self.document_count / df)

            for doc, tf in postings:
                if doc == doc_id:
                    tf_idf_weights[term] = (1 + math.log(tf)) * idf
                    break
        return tf_idf_weights


    def search_similar(self, doc_id: int, num_results: int):
        print("     now im searching for similar documents ....")
        tf_idf_weigths = self.calculate_tf_idf(doc_id)

        top_terms = sorted(tf_idf_weigths.items(), key=lambda x: x[1], reverse=True)[:10]
        terms = [term for term, score in top_terms]

        print("\nTHE MOST IMPORTANT TERMS IN A DOCUMENT: ")
        for t in terms:
            print(f"-- {t}")

        return self.search_tokenized(terms, top_k=num_results + 1)[1:]
