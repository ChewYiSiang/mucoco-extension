from llm_models.code_llms import CodeLLM
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class CodeReasoningLLM(CodeLLM):
    pass

class OpenAIReasoningLLM(CodeReasoningLLM):
    def __init__(self, model_name = 'gpt-5'):
        self.model_name = model_name
        self.client = OpenAI()

    def invoke(self, input_variables: Dict[str, str], prompt_template: str) -> str | None:
        prompt = prompt_template.format(**input_variables)
        result = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            reasoning={ "effort": "low" },
            text={ "verbosity": "low" },
        )
        return result.output_text

# class ClaudeReasoningLLM(CodeReasoningLLM):
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.client = anthropic.Anthropic()
    
#     def invoke(self, input_variables, prompt_template):
#         prompt = prompt_template.format(**input_variables)

#         message = self.client.messages.create(
#             model=self.model_name,
#             max_tokens=1000,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0
#         )

#         return message.content[0].text
    