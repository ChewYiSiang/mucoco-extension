from abc import ABC
import textwrap 
from typing import List, Dict
import random

class PromptTemplate(ABC):
    pass

class MCQPromptTemplate(PromptTemplate):
    def zero_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            Using this code snipper, answer the MCQ question below. 
            {question}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be? Only return your final answer below as A, B, C, D, or E.

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            Your answer: 
        """)
        return prompt

class OpenEndedPromptTemplate(PromptTemplate):
    def zero_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            # Complete the given code snippet using the description below. Only complete the code function and do not add any other details. If a helper function is given, return it together with your answer without modifying it.
            {task}
                                 
            # Your Answer: 
            {code}
        """)
        return prompt
    
    def one_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            # Complete the code for the following function given it's description. You may use the given example to write your code. Return your answer as a complete function. 
            {task}
                                 
            # Example:
            {example}
                                 
            # Your answer: 
            {code}

        """)
        return prompt
    
    def few_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            # Complete the code for the following function given it's description. You may use the given examples to write your code. Return your answer as a complete function.
            {task}

            # Examples:
            {example}
                                 
            # Your answer: 
            {code}
        """)
        return prompt

    def structure_few_shot_examples(test_cases: Dict[str, str]) -> str:
        return "\n".join(">>> " + test + "\n" + test_cases[test] for test in test_cases)
    
    def structure_one_shot_example(test_cases: Dict[str, str]) -> str:
        random_example = random.choice(list(test_cases.keys()))
        return ">>> " + random_example + "\n" + test_cases[random_example]
    
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

class LlamaOpenEndedPromptTemplate(OpenEndedPromptTemplate):
    def zero_shot_prompt():
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            # Complete the given code snippet using the description below. Only complete the code function and do not add any other details.
            {task}<|eot_id|>
                                 
            <|start_header_id|>user<|end_header_id|>
            # Code Snippet:
            {code}<|eot_id|>
                                 
            # Your Answer: 
            <|start_header_id|>assistant<|end_header_id|>
        """)
        return prompt

    def one_shot_prompt():
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            # Complete the code for the following function given it's description. You may use the given example to write your code. Return your answer as a complete function.
            {task}<|eot_id|>
                                 
            <|start_header_id|>user<|end_header_id|>
            # Code Snippet:
            {code}
                                 
            # Example:
            {example}<|eot_id|>
                                 
            # Your answer:
            <|start_header_id|>assistant<|end_header_id|>
        """)
        return prompt

    def few_shot_prompt():
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            # Complete the code for the following function given it's description. You may use the given examples to write your code. Return your answer as a complete function.
            {task}<|eot_id|>
                                 
            <|start_header_id|>user<|end_header_id|>
            # Code Snippet:
            {code}
                                 
            # Examples:
            {example}<|eot_id|>
                                 
            # Your answer:
            <|start_header_id|>assistant<|end_header_id|>
        """)
        return prompt


if __name__ == "__main__":
    x = {
        "func1": "abc",
        "func2": "def"
    }
    tc = OpenEndedPromptTemplate.structure_few_shot_examples(x)
    print(tc)