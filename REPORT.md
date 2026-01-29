# PROJECT REPORT

## Neural Reranker

### Getting started
The first step towards incorporating neural reranker was selecting the appropriate model. For that, we selected mmarco-mMiniLMv2-L12-H384-v1, which in the end showed good performance during the testings. The previous function for retrieving top-k documents was not changed so that it always retrieves top 100 documents by using the BM-25 algorithm, and then uses the neural reranker to select top 10, 5, etc documents (as user selected).

### GPU support
The main problem we encountered was that the process of reranking is much more computationally expensive, and therefore takes a lot of time to rerank top 100 documents (more than 10 seconds on CPU). However, by utilizing locally available GPU, such as RTX 3050, we managed to significantly reduce running time of reranker, between 1.4 and 1.9 seconds, more than 5 times faster than running on CPU. In order to do that, we had to simply install the newest CUDA for pytorch from their official website (we used CUDA 12.6 support for torch).

### Software architecture
Software architecture remained largely the same, the only major change for this part was creating a new class NeuralReranker, with corresponding methods for object construction and reranking. Reranker is then created and used inside the python script that handles routes for searching.



## RAG
The RAG part of the project was a lot more tricky, since it included using external LLM model that was not running locally on our PC (because models are much more complex to run, even for RTX 3050). The first model that we selected was groq, sepcifically "openai/gpt-oss-20b", which worked well, but showed huge restrictions when sending the prompts (for example, the TPM limit was 8000 tokens, which is virtually unusable when the document is larger, since we would have to either truncate the document that we send or send in batches). Therefore, in the end we selected the gemini models, alternating between gemini-2.5-flash-lite and gemini-2.5-flash.

The process of answering the question is done in 2 steps:
1) send a prompt to the LLM and ask him to analyze whether the user query is a question or not
2) if the query was a question, give the most relevant document to the LLM and make him try to answer the question (using only the sent document)

### Is the query a question?
This part was solved by sending the following prompt to the LLM 
```text
Respond only with 'yes' or 'no'. Is the following sentence a question? Sentence: '{query}' (Portuguese)" whery query is a user query entered into a search engine.
```

### What is the answer to question?
If the answer we receive from the LLM is positive, we send another prompt

```text
You are an assistant. Use only the following text to answer the question. 
If the answer is not in the text, respond with "Desculpe. Eu não sei."

Text:
{document}

Question:
{question}
```

Therefore, we provide an LLM with only the user query and the most relevant document (after neural reranking).

In order to incorporate this change in the software, we had to slightly modify the Document class in the model.py module, so that it also contains the "answer" atribute, which is equal to None in case that the user query was not a question.

The answer to the question is display directly below the search bar, between the search bar and the most relevant document (user has an option to remove the answer field, if he wants to).

## Additional AI enhancments
### Best snippet extraction
After the neural reranker selects the top document, we also extract the single paragraph that best matches the user’s query.
It works as following:
1. The document text is split into paragraphs using double line breaks (\n\n) as separators
2. Each paragraph is paired with the original query and scored using the same neural reranker model (mmarco-mMiniLMv2-L12-H384-v1) we use for document reranking
3. The paragraph with the highest relevance score is selected as the “best snippet”

This snippet is included in the search result and displayed in the UI, so users can switch their view between see the most relevant part of the document without having to scan through everything, and the whole document.

### User query improvement
Sometimes, it can happen that the user query is not completely correct, e.g. contains the grammar errors etc. In order to combat that, we once again used gemini to provide corrected query, but only as a suggestion, with user being able to then select the recommended query as his own query. Here, we do not restrict the LLM to the information it uses, we simply send him the users query with the following instructions
1. If the query is valid and understandable, respond exactly with the original query. 
2. If the query is unclear, incomplete, or poorly phrased, suggest a corrected and improved version of the query. 
3. Return only the query text. 
4. The output should be in Portuguese.
Improved query is then sent to the frontend and displayed, if it exists, with corresponding message "Voce pensou..."

The first iteration of this AI enhancement used another model run locally on our GPU https://huggingface.co/pierreguillou/gpt2-small-portuguese . The reason why it could have potentially been good was that it was trained on Portuguese wikipedia by using transfer learning, but it showed really terrible results, therefore making us abandon this experiment.

