import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class GemmaMCQPromptTemplate(PromptTemplate):
    """Gemma-specific prompt templates for MCQ tasks using Gemma chat template format with <start_of_turn> tokens."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <start_of_turn>user
            Using this code snippet, answer the MCQ question below. 
            {question}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be? Only return your final answer below as A, B, C, D, or E.

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            What is your answer?<end_of_turn>
            <start_of_turn>model
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        # MCQ tasks typically don't use one-shot prompting, but including for completeness
        return self.zero_shot_prompt()

    def few_shot_prompt(self) -> str:
        # MCQ tasks typically don't use few-shot prompting, but including for completeness
        return self.zero_shot_prompt()


class GemmaOpenEndedPromptTemplate(PromptTemplate):
    """Gemma-specific prompt templates for open-ended code generation tasks using Gemma chat template format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <start_of_turn>user
            Complete the given code snippet using the description below. 
            Complete the function body and do not add any explanations or extra text. 
            Return all code snippet in your answer. 
            Preserve indentation.

            ### Example:

            # Task
            Return a function that returns the largest integer in a list.

            # Code Snippet
            def find_max(nums):

            # Your Answer
            def find_max(nums):
                return max(nums)

            ### Task:
            {task}

            ### Code Snippet:
            {code}

            What is your answer?<end_of_turn>
            <start_of_turn>model
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <start_of_turn>user
            Complete the code for the following function given it's description. Only complete the code function and do not add any other details. You may use the given example to write your code. Return your answer as a complete function, including any provided code.

            {task}
                                 
            # Example:
            {example}
                                 
            # Code Snippet:
            {code}

            What is your answer?<end_of_turn>
            <start_of_turn>model
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <start_of_turn>user
            Complete the code for the following function given it's description. You may use the given examples to write your code. Return your answer as a complete function.

            {task}

            # Examples:
            {example}
                                 
            # Code Snippet:
            {code}

            What is your answer?<end_of_turn>
            <start_of_turn>model
        """)
        return prompt


if __name__ == "__main__":
    pass