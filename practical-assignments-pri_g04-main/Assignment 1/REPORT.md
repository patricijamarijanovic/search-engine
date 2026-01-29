# PROJECT REPORT

## Building inverted index

### Tokenizer
The first step towards building the inverted index (or indexer whose job is to produce the inverted and forward index) was to create suitable input data for the indexer. Or, to tokenize all the words to produce terms that should serve as key values in the inverted index. To achieve the most efficient text processing with the tokenizer, the following operation order was established:
```
    - set text to lowercase
    - remove URLs (if set as the operation to conduct)
    - remove emails (if set as the operation to conduct)
    - split the text into python list of tokens list[str]
    - separate alphanumeric tokens (if set as the operation to conduct)
    - remove numbers (fully numeric tokens, if set as the operation to conduct)
    - remove the tokens with the length less than threshold
    - removing the stopwords (if set as the operation of the conflict)
    - stemming of the words (if set as the operation to conduct)
```

Additionaly, in order to reduce the time neccessary to produce the inverted index, we have decided to use pre-compiled REGEXs for the URLs and e-mails, as 
well as storing 100_000 most recent stem terms by using lru_cache. Those modifications reduced the neccessary time to produce stemmed inverted index from 60 minutes to 40 minutes.

Default settings that we have chosen for our tokenizer were following:
```
    - lowercase True
    - remove URLs True
    - remove emails True
    - separate alphanumeric True
    - remove numbers True
    - token length threshold 1
    - stemmer True
    - stopwords False
```

### Indexer

In order to build the functional search engine, several .json or .jsonl files were neccessary to produce during the indexing proces
```
    - documents_metadata.jsonl
    - documents_stats.jsonl
    - final_index.jsonl
    - forward_index.db
    - indexer_metadata.jsonl
    - offset_index.json
```

"documents_metadata.jsonl" contains global metadata of the processed text documents, doc_count, total_tokens, avg_doc_length.  
"documents_stats.jsonl" stores pairs of values in the key:value format, where key is the document_id (doc_id), while value is the length of the corresponding document.  
"final_index.jsonl" stores our inverted index, in the format term: postings_list, where postings_list is the tuple of (doc_id, freq). In order to satisfy the given memory condition, we had to stream our dataset in batches, with batch size set to 750 (out of maximum value of 1000). This allowed us to keep the memory around 1400MB, low enough for the given constraint of 2GB. Additionaly, inverted index was created in 2 phases.
```
    - creating SPIMI blocks - in this phase we stored the index in smaller batches, named block_%d, sequentially as we passed our dataset. First we tokenized each document and stored its postings, before flushing each block to the disk in the format term: postings_list, where key was the term that appeared in the document, and posting contained the pair of document id and frequency of the term in the given document. Blocks were flushed to the disk either as we reached the memory threshold (set to 1800MB) or as certain number of tokens was stored.
    - merging SPIMI blocks - in this phase SPIMI blocks sorted alphabetically were merged into one final index, which was in the end flushed to the disk. After the merging was done, additionaly all temporary blocks are deleted from the disk.
```

"forward_index.db" was stored by using the sqlite3 library, as a normal database. Each rows contains given document_id, title and text columns. Like inverted index, it is created in batches (but this time with batch size set to 1000), to satisfy memory requirements. It is used to fetch document content when searching and giving text to the user.  
"indexer_metadata.jsonl" contains all the data about the indexer (all of its parameters). Therefore, it contains all the data passed to the tokenizer used by the indexer, as well as some aditional data:
```
    - file_path (path to the file that should be indexed)
    - min_term_frequency (minimum number that the term should appear to be stored in the inverted index)
    - output directory (path to directory where all .json and .db files are store)
    - format of the inverted index
```

Default value of the min_term_frequency is set to 5.  

"offset_index.json" this file contains another list of pairs term: offset_in_final_index.jsonl. In other words, for given term, it says WHAT is the offset from the start of the final_index.jsonl to the postings list of the term that is the key value. It was created because originally, searching for the postings list by simply using final_index[term] or final_index.get(term) was simply too slow and resulted in search queries of a minute. The main reason for that was the sheer volume of the inverted index file (more than 2.5GB). Offset index has a much smaller size of around 13MB, therefore making the searching much faster.

### Searcher

Searcher is used to process the user query -> to retrieve relevant documents from the inverted index (SearchEngine class takes an index path as parameter for initialization) and to rank documents using bm25 score. To ensure efficiency and stay within the memory limits, we decided not to load the entire index into memory, but instead use the offset index (offset_index.json) to locate terms dynamically when needed. The offset_index.json file stores the byte offset of each term inside final_index.jsonl. When a query term appears, the searcher looks up its offset, seeks directly to that position in the final_index.jsonl file, and retrieves the corresponding posting list without loading everything into memory.

Queries are processed using the same Tokenizer that was used during indexing.
The tokenization parameters are stored in indexer_metadata.jsonl, allowing the searcher to apply identical preprocessing steps when tokenizing user queries.
After tokenization, for each query term, the corresponding offset is retrieved from offset_index.json, and the posting list is read from final_index.jsonl.
Once all relevant postings are collected, documents are ranked using the BM25 score and sorted accordingly. 

For the Search Similar option, the selected document itself is used as the basis for the query.
Since directly using the entire document would result in an overly long and inefficient query, we first calculate the tf-idf score for every term within the document.
The top terms with the highest tf-idf scores are then selected to represent the document’s most significant keywords, and these terms are used as a query.
After that, the BM25-based search is performed as in the standard case.