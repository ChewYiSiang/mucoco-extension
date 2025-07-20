from abc import ABC
import textwrap 
from typing import List, Dict
import random
from code_generation.prompt_templates.prompt_template import PromptTemplate

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
    
    def one_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code, the input and a single example. Return the expected output in your answer. You may use the example to determine the expected output.
            
            # Your answer should only contain the expected output with no additional information. 
                                 
            {qn_desc}
                         
            # Code Snippet
            {full_sol}
            
            # Input
            {test_input}
                                 
            # Example
            {example}
                                 
            # Expected Output: 
        """)
        return prompt
    
    def few_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code, the input and a few examples. Return the expected output in your answer. You may use the examples to determine the expected output.
            
            # Your answer should only contain the expected output with no additional information. 
                                
            {qn_desc}
                        
            # Code Snippet
            {full_sol}
            
            # Input
            {test_input}
                                
            # Examples
            {example}
                                
            # Expected Output: 
        """)
        return prompt


if __name__ == "__main__":
    pass