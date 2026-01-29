import gc
import json
import os
import psutil
import glob
import heapq
import sqlite3
import pyarrow.dataset as ds
from collections import defaultdict
from time import time
from contextlib import ExitStack

from sapien.core.tokenizer import Tokenizer



class Indexer:
    def __init__(
        self,
        file_path: str,
        min_term_freq: int = 5,
        output_directory: str = "output",
        separate_alphanumeric: bool | int = False,
        remove_numbers: bool | int = False,
        remove_URLs: bool | int = False,
        remove_emails: bool | int = False,
        min_token_length: int = 1,
        lowercase: bool | int = False,
        stemmer: bool | int = False,
        stopwords: bool | int = False,
    ):
        
        # --- indexer parameters ---
        self.file_path: str = file_path
        self.min_term_freq: int = min_term_freq
        self.output_directory: str = output_directory
        self.inverted_format: str = "json"
        os.makedirs(self.output_directory, exist_ok=True)
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.block_count: int = 0
        self.token_count: int = 0
        self.token_threshold: int = 5000000  
        self.total_tokens: int = 0

        self.doc_lengths: dict[int, int] = {}
        self.doc_count: int = 0
        self.documents_stats_path: str = os.path.join(self.output_directory, "documents_stats.jsonl")
        self.database_path: str = "output\\forward_index.db"

        # --- tokenizer parameters ---
        tokenizer_params = {
            "separate_alphanumeric": separate_alphanumeric,
            "remove_numbers": remove_numbers,
            "remove_URLs": remove_URLs,
            "remove_emails": remove_emails,
            "min_token_length": min_token_length,
            "lowercase": lowercase,
            "stemmer": stemmer,
            "use_stopwords": stopwords,
        }
        self.tokenizer = Tokenizer(**tokenizer_params) #tokenizer is set as "property" of indexer
        self.current_process = psutil.Process(os.getpid())

        self.metadata = {
            "file_path": file_path,
            "min_term_freq": min_term_freq,
            "output_directory": output_directory,
            "inverted_format": "json",
            "separate_alphanumeric": bool(separate_alphanumeric),
            "remove_numbers": bool(remove_numbers),
            "remove_URLs": bool(remove_URLs),
            "remove_emails": bool(remove_emails),
            "min_token_length": min_token_length,
            "lowercase": bool(lowercase),
            "stemmer": bool(stemmer),
            "stopwords": bool(stopwords),
        }


    
    def output_configuration(self) -> str:
        '''
            Method that outputs the configuration of the indexer in the string format
            output - string
        '''
        indexer_configs = (
            f"Configuration:\n"
            f"  · Min term frequency: {self.min_term_freq}\n"
            f"  · Output directory: {self.output_directory}\n"
            f"  · Inverted index format: {self.inverted_format}\n"
        )
        tokenizer_configs = self.tokenizer.output_configuration()
        return indexer_configs + tokenizer_configs


    def store_metadata(self):
        '''
            Method that stores all indexer metadata in .jsonl file.
        '''
        path = os.path.join(self.output_directory, "indexer_metadata.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)


    def create_inverted_index(self):
        '''
            Method that loads the given dataset and creates corresponding inverted index, storing it on the disk
            Batch size is set to 750, since 1000 causes memory crashes, > 2000MB
            Inverted index is store in form term: postings, where each posting is (document_id, freqeuncy)
        '''
        dataset: ds.FileSystemDataset = self._load_dataset()

        document_id: int = 0
        skipped_documents: int = 0
        for i, batch in enumerate(dataset.to_batches(batch_size=750)):
            batch_start_time: float = time()
            text_col = batch.column("text") # type: ignore

            print(f"\nProcessing batch {i + 1} with {batch.num_rows} rows")
            for j, value in enumerate(text_col): # type: ignore
                text: str = value.as_py() # type: ignore
                document_id += 1
                if not text or not text.strip(): # type: ignore
                    skipped_documents += 1
                    continue

                tokens = self.tokenizer.tokenize(text) # type: ignore
                self._add_document(document_id, tokens)

            batch_end_time: float = time()
            print(f"Batch {i + 1} processed in {batch_end_time - batch_start_time:.2f} seconds")

        self._finalize() #store the final block
        del dataset # clear the memory
        gc.collect() 
        self._merge_blocks() # merge blocks 
        self._create_offset_index() # create "offset index"


    def create_forward_index(self):
        '''
            Method for creating and storing on disk the forward index. Unlike the inverted index,
            forward index is stored as database.
            Each row is (document_id, title, text)
            Since here less memory is used, batch size can be set to 1000 (maximum value), without crashing.
        '''
        dataset = self._load_dataset()

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                title TEXT,
                text TEXT
            )
        """)
        conn.commit()

        document_id: int = 0
        skipped_documents: int = 0

        for i, batch in enumerate(dataset.to_batches(batch_size=1000)):
            batch_start_time = time()

            title_column = batch.column("title") # type: ignore
            text_column = batch.column("text") # type: ignore

            print(f"\nProcessing batch {i + 1} with {batch.num_rows} rows")

            for j in range(batch.num_rows):
                title: str = title_column[j].as_py() # type: ignore
                text: str = text_column[j].as_py() # type: ignore
                document_id += 1

                if not text or not text.strip():
                    skipped_documents += 1
                    continue

                cursor.execute(
                    "INSERT INTO documents (doc_id, title, text) VALUES (?, ?, ?)",
                    (document_id, title, text)
                )

            conn.commit()

            batch_end_time: float = time()
            print(f"Batch {i + 1} processed in {batch_end_time - batch_start_time:.2f} seconds")

        conn.close() # disconnect from the database 

        del dataset, title_column, text_column, title, text # clear the memory
        gc.collect()
        print(f"Forward index created. Total documents: {document_id}, skipped: {skipped_documents}")


    def _load_dataset(self) -> ds.FileSystemDataset:
        '''
            Private method that loads the dataset from the given path.
            output: dataset to parse
        '''
        return ds.dataset(self.file_path, format="arrow")


    def _add_document(self, doc_id: int, tokens: list[str]):
        '''
            For the given document, count each term's frequency and add it to inverted index (current SPIMI block)
            input: doc_id -> currently parsed document's id
                   tokens -> list of tokens for the current document
        '''
        term_freq: dict[str, int] = defaultdict(int)
        valid_tokens: int = 0

        for term in tokens: # pass all tokens and store their frequency
            term_freq[term] += 1
            valid_tokens += 1
            self.total_tokens += 1

        self.doc_lengths[doc_id] = valid_tokens # store document length to disk
        self.doc_count += 1
        with open(self.documents_stats_path, "a", encoding="utf-8") as stats_file:
            stats_file.write(json.dumps({"doc_id": doc_id, "length": valid_tokens}) + "\n")

        for term, freq in term_freq.items(): # add postings to current inverted index block
            self.inverted_index[term].append((doc_id, freq))
            self.token_count += 1

        process = psutil.Process()
        memory_limit: int = 2 * 1024 * 1024 * 1024
        mem_usage = process.memory_info().rss

        if self.token_count >= self.token_threshold or mem_usage > memory_limit * 0.8: #if token threshold is passed, or memory limit, store index to disk
            self._write_block()
            self.token_count = 0


    def _write_block(self):
        '''
            Write the current index block to the disk in .jsonl format
        '''
        self.block_count += 1
        block_path = os.path.join(self.output_directory, f"block_{self.block_count}.jsonl")
        sorted_index = dict(sorted(self.inverted_index.items()))

        with open(block_path, "w", encoding="utf-8") as f:
            for term, postings in sorted_index.items():
                f.write(json.dumps({term: postings}) + "\n")

        print(
            f"SPIMI wrote block {self.block_count} "
            f"with {len(self.inverted_index)} terms, {block_path}"
        )
        del sorted_index # clear the memory for additional space before continuing indexing
        self.inverted_index.clear()
        gc.collect()
        self.token_count = 0


    def _finalize(self):
        '''
            Private method for storing the last index block to the disk, if it is not empty
        '''
        if self.inverted_index:
            self._write_block()

        print("All SPIMI blocks stored to the disk.")
        avg_doc_length = self.total_tokens / self.doc_count if self.doc_count > 0 else 0

        metadata = {
            "doc_count": self.doc_count,
            "total_tokens": self.total_tokens,
            "avg_doc_length": avg_doc_length
        }

        meta_path = os.path.join(self.output_directory, "documents_metadata.jsonl") # storing documents metadata to the disk in jsonl format
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Documents Metadata saved to {meta_path}")


    def _clear_memory_before_merge(self):
        '''
            Private method that clears all possible memory before merging the blocks (final phase of SPIMI indexing)
        '''
        if hasattr(self, "inverted_index"):
            self.inverted_index.clear()
            del self.inverted_index
            self.inverted_index = defaultdict(list)
        
        if hasattr(self, "doc_lengths"):
            self.doc_lengths.clear()
            del self.doc_lengths
            self.doc_lengths = {}

        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()
        
        process = psutil.Process(os.getpid())
        print(f"Memory after clearing: {process.memory_info().rss / 1024**2:.2f} MB")
    

    def _merge_blocks(self):
        '''
            Private method that merges all SPIMI blocks into one final inverted index and stores it on disk
        '''
        self._clear_memory_before_merge() # first we clear the memory from creating blocks
        
        block_files = sorted( # fetch all needed blocks, without metadata files
            f
            for f in glob.glob(os.path.join(self.output_directory, "*.jsonl"))
            if not f.endswith(("final_index.jsonl", "documents_stats.jsonl", "metadata.jsonl", "documents_metadata.jsonl"))
        )

        if not block_files: # no blocks to merge, should never happen
            print("No blocks to merge")
            return

        process = psutil.Process()
        memory_limit = 2 * 1024 * 1024 * 1024  # 2GB

        with ExitStack() as stack:
            block_handles = [
                stack.enter_context(open(path, encoding="utf-8")) for path in block_files
            ]
            block_iterators = [self._line_iterator(f) for f in block_handles]

            heap = []
            for block_id, iterator in enumerate(block_iterators):
                try:
                    term, postings = next(iterator)
                    heapq.heappush(heap, (term, postings, block_id))
                except StopIteration:
                    continue

            final_index_path = os.path.join(self.output_directory, "final_index.jsonl")
            temp_index_path = os.path.join(self.output_directory, "temp_index.jsonl")

            current_term = None
            current_postings = []
            term_count = 0

            with open(temp_index_path, "w", encoding="utf-8") as f:
                while heap:
                    mem_usage = process.memory_info().rss
                    if mem_usage > memory_limit * 0.9:
                        print(f"Memory usage high ({mem_usage / (1024 ** 2):.2f} MB), flushing...")

                    term, postings, block_id = heapq.heappop(heap)

                    if term == current_term:
                        current_postings.extend(postings)
                    else:
                        if current_term:
                            merged_postings = self._merge_postings(current_postings)
                            if len(merged_postings) >= self.min_term_freq:
                                f.write(json.dumps({current_term: merged_postings}) + "\n")
                                term_count += 1

                        current_term = term
                        current_postings = postings

                    try:
                        next_term, next_postings = next(block_iterators[block_id])
                        heapq.heappush(heap, (next_term, next_postings, block_id))
                    except StopIteration:
                        continue

                if current_term:
                    merged_postings = self._merge_postings(current_postings)
                    if len(merged_postings) >= self.min_term_freq:
                        f.write(json.dumps({current_term: merged_postings}) + "\n")
                        term_count += 1

        os.rename(temp_index_path, final_index_path)
        print(f"Merged {len(block_files)} blocks, {term_count} terms written to {final_index_path}")
        self._delete_temporary_blocks()


    def _delete_temporary_blocks(self):
        '''
            Private methods that, after merging all the blocks, deletes all temporary blocks.
        '''
        pattern = os.path.join(self.output_directory, "block_*.jsonl")
        files_to_delete = glob.glob(pattern)

        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

        print(f"Deleted {len(files_to_delete)} temporary blocks.")


    def _create_offset_index(self):
        '''
            Private method that creates offset index, in order to fetch postings list much faster for the given term.
            Each member is term: offset, where "offset" is the number of bytes from start of final_index.json where
            postings for the given terms are located. Works much faster than simples search by key. 
        '''
        final_index_path = os.path.join(self.output_directory, "final_index.jsonl")
        offset_index_path = os.path.join(self.output_directory, "offset_index.json")

        offsets = {}
        with open(final_index_path, "rb") as f: 
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    term = next(iter(json.loads(line).keys()))
                    offsets[term] = pos
                except Exception as e:
                    print(f"Failed to read line at {pos}: {e}")
                    continue

        with open(offset_index_path, "w", encoding="utf-8") as out:
            json.dump(offsets, out, indent=2, ensure_ascii=False)

        print(f"Offset index built: {len(offsets)} terms -> {offset_index_path}")


    @staticmethod
    def _merge_postings(postings: tuple[int, int]) -> list[tuple[int, int]]:
        '''
            Combines duplicate (doc_id, frequency) postings into a single posting for given doc_id
        '''
        merged: dict[int, int] = defaultdict(int)
        for doc_id, freq in postings:
            merged[doc_id] += freq
        return sorted(merged.items())


    @staticmethod
    def _line_iterator(file_handle):
        '''
            generator that yields (term, postings) from the .jsonl blocks.
        '''
        for line in file_handle:
            if not line.strip():
                continue
            data = json.loads(line)
            term, postings = next(iter(data.items()))
            yield term, postings

