import os
import textwrap
from huggingface_hub import login
from langchain_mistralai.chat_models import ChatMistralAI
from huggingface_hub import InferenceClient
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from mistralai import Mistral
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class CodeLLM(ABC):
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        pass

    @abstractmethod
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        pass

class MistralSmall(CodeLLM):
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
    
class Codestral(CodeLLM):
    def __init__(self, model_name: str = "codestral-latest"):
        self.model_name = model_name
        self.client = Mistral(api_key=os.environ["CODESTRAL_API_KEY"])

    
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        prompt = prompt_template.format(**input_variables)
        message = [{"role": "user", "content": f"{prompt}"}]
        result = self.client.chat.complete(
            model=self.model_name,
            messages=message,
            temperature=0
        )
        return result.choices[0].message.content


class DeepSeekLLM(CodeLLM):
    def __init__(self, model_name: str = 'deepseek-chat'):
        self.model_name = model_name
        self.client = OpenAI(api_key = os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    def return_system_prompt(self) -> str:
        system_prompt = """"""
        return system_prompt
    
    def invoke(self, input_variables, prompt_template):
        prompt = prompt_template.format(**input_variables)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.return_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content

class MistralGPU(CodeLLM):
    def __init__(self, model_name):
        super().__init__(model_name)

