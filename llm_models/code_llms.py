import os
import textwrap
from huggingface_hub import login
from langchain_mistralai.chat_models import ChatMistralAI
from huggingface_hub import InferenceClient
from typing import Dict
from langchain.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI

class CodeLLM(ABC):
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        pass

    @abstractmethod
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        pass

class Mistral(CodeLLM):
    def __init__(self, model_name: str = "mistral-small-latest") -> ChatMistralAI | None:
        self.model_name = model_name
        try:
            self.model = ChatMistralAI(
                api_key = os.environ["MISTRAL_API_KEY"],
                model=self.model_name,
                temperature= 0
            )

        except KeyError as e:
            print("Mistral API key could not be obtained from .env")
            return None        
                        
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("human", prompt_template)
            ]
        )
        chain = prompt | self.model
        ans = chain.invoke(input = input_variables)
        return ans.content
    
class OpenAILLM(CodeLLM):
    def __init__(self, model_name : str = 'gpt-4o'):
        self.model_name = model_name
        self.client = OpenAI()
    
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        prompt = prompt_template.format(**input_variables)
        result = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0,
        )
        return result.output_text
    
class DeepSeekLLM(CodeLLM):
    def __init__(self, model_name: str = 'deepseek-chat'):
        self.model_name = model_name
        self.client = OpenAI(api_key = os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    def return_system_prompt(self) -> str:
        system_prompt = """You are a coding assistant. 
    Follow the task strictly:
    - Only complete the given code snippet based on the task description. 
    - Do not output explanations, comments, or extra text unless explicitly part of the code. 
    - Always return valid Python code. 
    - Preserve indentation exactly as in the snippet provided.
    - Do not wrap the code in Markdown fences (```)."""
        return system_prompt
    
    def invoke(self, input_variables, prompt_template):
        prompt = prompt_template.format(**input_variables)

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.return_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

class MistralGPU(CodeLLM):
    def __init__(self, model_name):
        super().__init__(model_name)

if __name__ == "__main__":
    load_dotenv()
    llm = DeepSeekLLM()
    x = llm.invoke({"d": """# You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 


# Code Snippet
def f(w):
    ls = list(w)
    omw = ''
    while len(ls) > 0 + 0:
        omw += ls.pop(0 + 0)
        if len(ls) * (1 + 1) > len(w):
            return w[len(ls):] == omw
    return 0 + 0

# Input
"flak"

# Expected Output:                      
### Your answer"""}, "{d}")
    print(x)