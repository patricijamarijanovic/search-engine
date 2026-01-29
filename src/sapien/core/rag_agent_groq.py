import os 
from dotenv import load_dotenv
from groq import Groq


class RagAgentGroq:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY nije postavljen u varijabli okoline!")
        
        #print(f"Groq api key: {self.groq_api_key}")

        self.groq_client = Groq(api_key=self.groq_api_key)

    
    def check_if_its_question(self, query: str) -> str:
        prompt = f"Respond only with 'yes' or 'no'. Is the following sentence a question? Sentence: '{query}' (Portuguese)"

        completion = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )

        answer = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                answer += content

        return answer
    

    def answer_question_with_document(self, question: str, document: str):
        prompt = f"""
                You are an assistant. Use only the following text to answer the question. 
                If the answer is not in the text, respond with 'Desculpe. Eu não sei.'

                Text:
                {document}

                Question:
                {question}
                """

        print(f"Length of document: {len(document)}.")
        completion = self.groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # deterministički odgovor
                    max_completion_tokens=8192,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=True
                )

        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""

        return answer