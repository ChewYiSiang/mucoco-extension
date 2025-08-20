import os
from huggingface_hub import login
from langchain_mistralai.chat_models import ChatMistralAI
from huggingface_hub import InferenceClient
from typing import Dict
from langchain.prompts import ChatPromptTemplate
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv

class CodeLLM(ABC):
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        pass

    @abstractmethod
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        pass

    @staticmethod
    def process_ans(text: str) -> str:
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("No code block found")

class MetaLlama(CodeLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct") -> ChatMistralAI | None:
        hf_token = os.environ["llama_hf_API"]
        login(token=hf_token)

        self.model_name = model_name

        try:
            self.model = InferenceClient(
                provider='hyperbolic',
                api_key=hf_token,
            )
        except KeyError as e:
            if type(e) == KeyError:
                print("Llama API key could not be obtained from .env")
            else:
                print("Could not deploy llama model due to the following error: {e}".format(e = e))
    
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        prompt = prompt_template.format(**input_variables)
        ans = self.model.chat.completions.create(
            model = self.model_name,
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature= 0
        )        
        return ans.choices[0].message.content

    @staticmethod
    def process_ans(text: str) -> str:
        return CodeLLM.process_ans(text)

class Mistral(CodeLLM):
    def __init__(self, model_name: str = "mistral-small-latest") -> ChatMistralAI | None:
        self.model_name = model_name
        try:
            self.model = ChatMistralAI(
                api_key = os.environ["mistral_API"],
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
    
    @staticmethod
    def process_ans(text: str) -> str:
        return CodeLLM.process_ans(text)


if __name__ == "__main__":
    load_dotenv()
    llm = Mistral()
    text, tokens, avg_lp = llm.invoke({"x": "hi"}, "Say hello to {x} in one short sentence.")
    print(text)