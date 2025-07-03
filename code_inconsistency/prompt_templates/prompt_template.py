from abc import ABC
import textwrap 
from typing import List, Dict
import random

class PromptTemplate(ABC):
    pass

class CodeInconsistencyPromptTemplate(PromptTemplate):
    def zero_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 
            {qn_desc}
                         
            # Code Snippet
            {full_sol}
            
            # Input
            {test_input}
                                 
            # Expected Output: 
        """)
        return prompt


if __name__ == "__main__":
    pass