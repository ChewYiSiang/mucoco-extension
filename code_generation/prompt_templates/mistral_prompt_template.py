import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class MistralMCQPromptTemplate(PromptTemplate):
    """Mistral-specific prompt templates for MCQ tasks using Mistral chat template format with [INST] tokens."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <s>[INST] Using this code snippet, answer the MCQ question below. 
            {question}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be? Only return your final answer below as A, B, C, D, or E.

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            What is your answer? [/INST]
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        # MCQ tasks typically don't use one-shot prompting, but including for completeness
        return self.zero_shot_prompt()

    def few_shot_prompt(self) -> str:
        # MCQ tasks typically don't use few-shot prompting, but including for completeness
        return self.zero_shot_prompt()


class MistralOpenEndedPromptTemplate(PromptTemplate):
    """Mistral-specific prompt templates for open-ended code generation tasks using Mistral chat template format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <s>[INST] Complete the given code snippet using the description below. 
            Complete the function body and do not add any explanations or extra text. 
            Return the complete python program and any necessary import statements.
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

            What is your answer? [/INST]
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <s>[INST] Complete the given code snippet using the description below. 
            Complete the function body and do not add any explanations or extra text. 
            Return the complete python program and any necessary import statements.
            Preserve indentation.

            {task}
                                 
            # Example:
            {example}
                                 
            # Code Snippet:
            {code}

            What is your answer? [/INST]
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <s>[INST] Complete the given code snippet using the description below. 
            Complete the function body and do not add any explanations or extra text. 
            Return the complete python program and any necessary import statements.
            Preserve indentation.

            {task}

            # Examples:
            {example}
                                 
            # Code Snippet:
            {code}

            What is your answer? [/INST]
        """)
        return prompt


if __name__ == "__main__":
    pass