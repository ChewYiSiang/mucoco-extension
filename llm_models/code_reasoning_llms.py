from llm_models.code_llms import CodeLLM
from typing import Dict
import anthropic
from dotenv import load_dotenv
import os
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


class DeepSeekReasonerLLM(CodeReasoningLLM):
    def __init__(self, model_name: str = 'deepseek-reasoner'):
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

class ClaudeReasoningLLM(CodeReasoningLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = anthropic.Anthropic()
    
    def invoke(self, input_variables, prompt_template):
        prompt = prompt_template.format(**input_variables)

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return message.content[0].text
    

if __name__ == "__main__":
    load_dotenv()
    l = ClaudeReasoningLLM("claude-sonnet-4-5-20250929")
    l.invoke({"d": "What's 1 + 2?"}, "{d}")