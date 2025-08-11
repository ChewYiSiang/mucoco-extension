from abc import ABC
import textwrap 
from typing import List, Dict
import random

class PromptTemplate(ABC):

    @staticmethod
    def structure_few_shot_examples(test_cases: Dict[str, str]) -> str:
        """
        This function is used to format examples in a dictionary into a standardised doctest format to be included in a few shot prompt template. This is used to structure the examples for few shot prompts.

        Args: 
            test_cases (Dict[str, str]): a dictionary containing examples with inputs and their respective expected outputs. 

                The dictionary keys are function calls with inputs. The value for the dicionary keys is the expected output from executing the function and input.

        Returns:
            str: a string representing the structured few shot examples
        """
        return "\n".join(">>> " + test + "\n" + test_cases[test] for test in test_cases)
    
    @staticmethod
    def structure_one_shot_example(test_cases: Dict[str, str] | str) -> str:
        """
        This function is used to format examples in a dictionary into a standardised doctest format to be included in a one shot prompt template. This is used to structure the examples used for one shot prompts.

        Args: 
            test_cases (Dict[str, str]): a dictionary containing examples with inputs and their respective expected outputs. 

                The dictionary keys are function calls with inputs. The value for the dicionary keys is the expected output from executing the function and input.

        Returns:
            str: a string representing the structured one shot examples
        """

        if isinstance(test_cases, Dict):
            random_example = random.choice(list(test_cases.keys()))
            return ">>> " + random_example + "\n" + test_cases[random_example]
        else:
            return test_cases

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
            # Complete the code for the following function given it's description. Only complete the code function and do not add any other details. You may use the given example to write your code. Return your answer as a complete function, including any provided code. 
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