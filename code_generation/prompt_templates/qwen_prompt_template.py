import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class QwenOpenEndedPromptTemplate(PromptTemplate):
    """Qwen-specific code generation prompt templates using ChatML format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Complete the given code snippet using the description below. Complete the function body and do not add any explanations or extra text. Return all code snippet in your answer. Preserve indentation.

            Rules:
            - Return ONLY the complete function code
            - No explanations, comments, or additional text
            - Preserve exact indentation and formatting
            - Include the complete function with the provided signature<|im_end|>
            <|im_start|>user
            ### Task:
            {task}

            ### Code Snippet:
            {code}

            Complete this function.<|im_end|>
            <|im_start|>assistant
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Complete the code for the following function given its description. Only complete the code function and do not add any other details. Use the given example to write your code. Return your answer as a complete function, including any provided code.

            Rules:
            - Return ONLY the complete function code
            - No explanations, comments, or additional text
            - Use the provided example to understand the pattern
            - Preserve exact indentation and formatting<|im_end|>
            <|im_start|>user
            {task}
                                 
            # Example:
            {example}
                                 
            # Code Snippet:
            {code}

            Complete this function.<|im_end|>
            <|im_start|>assistant
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Complete the code for the following function given its description. Use the given examples to write your code. Return your answer as a complete function.

            Rules:
            - Return ONLY the complete function code
            - No explanations, comments, or additional text
            - Use the provided examples to understand the pattern
            - Preserve exact indentation and formatting<|im_end|>
            <|im_start|>user
            {task}

            # Examples:
            {example}
                                 
            # Code Snippet:
            {code}

            Complete this function.<|im_end|>
            <|im_start|>assistant
        """)
        return prompt


class QwenThinkingOpenEndedPromptTemplate(PromptTemplate):
    """Qwen-3 specific code generation prompt templates with thinking mode enabled."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Complete the given code snippet using the description below. Think through the problem step by step, then provide the complete function code. Do not add any explanations or extra text in your final answer.

            Rules:
            - Think through the problem in <think></think> tags
            - Return ONLY the complete function code after thinking
            - No explanations in the final answer
            - Preserve exact indentation and formatting<|im_end|>
            <|im_start|>user
            ### Task:
            {task}

            ### Code Snippet:
            {code}

            Complete this function.<|im_end|>
            <|im_start|>assistant
            <think>
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Complete the code for the following function given its description. Think through the problem step by step, then provide the complete function code. Use the given example to understand the pattern.

            Rules:
            - Think through the problem in <think></think> tags
            - Return ONLY the complete function code after thinking
            - Use the provided example to understand the pattern
            - No explanations in the final answer<|im_end|>
            <|im_start|>user
            {task}
                                 
            # Example:
            {example}
                                 
            # Code Snippet:
            {code}

            Complete this function.<|im_end|>
            <|im_start|>assistant
            <think>
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Complete the code for the following function given its description. Think through the problem step by step, then provide the complete function code. Use the given examples to understand the pattern.

            Rules:
            - Think through the problem in <think></think> tags
            - Return ONLY the complete function code after thinking
            - Use the provided examples to understand the pattern
            - No explanations in the final answer<|im_end|>
            <|im_start|>user
            {task}

            # Examples:
            {example}
                                 
            # Code Snippet:
            {code}

            Complete this function.<|im_end|>
            <|im_start|>assistant
            <think>
        """)
        return prompt


class QwenMCQPromptTemplate(PromptTemplate):
    """Qwen-specific MCQ prompt templates using ChatML format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Using this code snippet, answer the MCQ question below. If this code is run with the test case, determine what the answer will be.

            Rules:
            - Answer ONLY with the letter (A, B, C, D, or E)
            - No explanations, comments, or additional text
            - Analyze the code execution carefully<|im_end|>
            <|im_start|>user
            {question}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be?

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            Your answer:<|im_end|>
            <|im_start|>assistant
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Using this code snippet, answer the MCQ question below. Use the provided example to understand the pattern.

            Rules:
            - Answer ONLY with the letter (A, B, C, D, or E)
            - No explanations, comments, or additional text
            - Use the example to understand the expected format<|im_end|>
            <|im_start|>user
            {question}

            # Example:
            {example}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be?

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            Your answer:<|im_end|>
            <|im_start|>assistant
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            Using this code snippet, answer the MCQ question below. Use the provided examples to understand the pattern.

            Rules:
            - Answer ONLY with the letter (A, B, C, D, or E)
            - No explanations, comments, or additional text
            - Use the examples to understand the expected format<|im_end|>
            <|im_start|>user
            {question}

            # Examples:
            {example}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be?

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            Your answer:<|im_end|>
            <|im_start|>assistant
        """)
        return prompt


if __name__ == "__main__":
    pass