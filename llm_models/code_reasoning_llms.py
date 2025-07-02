from abc import abstractmethod
from langchain_community.chat_models import ChatOpenAI
from llm_models.code_llms import CodeLLM
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import string
import ast


import os

class CodeReasoningLLM(CodeLLM):
    @abstractmethod
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        pass

    @abstractmethod
    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        pass

    @abstractmethod
    def process_ans(text: str) -> str:
        pass



class DeepSeekLLM(CodeLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if self.model_name == "DeepSeekR1":
            try:
                self.model = ChatOpenAI(
                    openai_api_key= os.getenv("openrouterDeepSeekR1_API"),
                    openai_api_base= os.getenv("openrouterBaseURL"),
                    model_name= os.getenv("openrouterDeepSeekR1")
                )
            except Exception as e:
                raise Exception("Could not launch model due to the following error: ".format(e = e))
        else:
            raise Exception("Unknown DeepSeek model")     

    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        formatter = string.Formatter()
        prompt_variables = [fname for _, fname, _, _ in formatter.parse(prompt_template) if fname]
        new_prompt = PromptTemplate(input_variables=prompt_variables, template=prompt_template)

        llm_chain = LLMChain(prompt = new_prompt, llm = self.model)
        ans = llm_chain.run(input_variables)

        return ans
    
    @staticmethod
    def process_ans(source_code: str) -> str:
        tree = ast.parse(source_code)
        lines = source_code.strip().splitlines()
        func_name_preserved = False
        lines_to_remove = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_name_preserved = True
            elif isinstance(node, ast.ImportFrom):
                pass
            else:
                for i in range(node.lineno - 1, node.end_lineno):
                    lines_to_remove.add(i)
        
        processed_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]

        return "\n".join(processed_lines)
                    
    def zero_shot_test(self):
        pass

    def one_shot_test(self):
        pass

    def few_shot_test(self):
        pass
