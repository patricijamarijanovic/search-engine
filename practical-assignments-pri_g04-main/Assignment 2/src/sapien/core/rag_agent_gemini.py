import os 
from dotenv import load_dotenv
from google import genai


class RagAgentGemini:
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY nije postavljen u varijabli okoline!")
        
        #print(f"Groq api key: {self.groq_api_key}")

        self.gemini_client = genai.Client()
        self.model = "gemini-2.5-flash-lite"

        #response = self.gemini_client.models.generate_content(
        #    model=self.model,
        #   contents="Explain how AI works in a few words"
        #)


    def check_if_its_question(self, query: str) -> str:
        prompt = f"Respond only with 'yes' or 'no'. Is the following sentence a question? Sentence: '{query}' (Portuguese)"

        answer = self.gemini_client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        print(answer.text)
        return answer.text
    

    def answer_question_with_document(self, question: str, document: str):
        prompt = f"""
                You are an assistant. Use only the following text to answer the question. 
                If the answer is not in the text, respond with 'Desculpe. Eu não sei.'

                Text:
                {document}

                Question:
                {question}
                """

        answer = self.gemini_client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return answer.text
    

    def improve_query_if_needed(self, query: str) -> str | None:
        prompt = f"""
        You are a query improvement assistant. 
        Your task is to review a user's search query. 

        1. If the query is valid and understandable, respond exactly with the original query. 
        2. If the query is unclear, incomplete, or poorly phrased, suggest a corrected and improved version of the query. 
        3. Return only the query text. 
        4. The output should be in Portuguese.

        Example:
        User query: "Cristiano Ronaldo stat"
        Output: "Estatísticas de Cristiano Ronaldo"

        User query: "Clima em Lisboa"
        Output: "Clima em Lisboa"

        Now process the following user query:
        "{query}"
        """

        response = self.gemini_client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        improved_query = response.text.strip() 

        if query.lower() != improved_query.lower():
            return improved_query
        
        return None
