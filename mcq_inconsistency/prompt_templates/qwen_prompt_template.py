import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class QwenMCQInconsistencyPromptTemplate(PromptTemplate):
    """Qwen-specific MCQ inconsistency prompt templates using ChatML format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            You are given a code snippet, a description of the code and several choices. Choose the correct option that completes the code snippet based on the question description.

            Rules:
            - Answer ONLY with the alphabet corresponding to the option (A, B, C, etc.)
            - No explanations, comments, or additional text
            - Choose the most appropriate option based on the code and description<|im_end|>
            <|im_start|>user
            {qn_desc}
                        
            # Code Snippet
            {task}
            
            # Choices
            {choices}
                                
            Which option correctly completes the code snippet?<|im_end|>
            <|im_start|>assistant
        """)
        return prompt
        
    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            You are given a code snippet, a description of the code, an example and several choices. Choose the correct option that completes the code snippet based on the question description.

            Rules:
            - Answer ONLY with the alphabet corresponding to the option (A, B, C, etc.)
            - No explanations, comments, or additional text
            - Use the provided example to understand the pattern<|im_end|>
            <|im_start|>user
            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Example
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?<|im_end|>
            <|im_start|>assistant
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            You are given a code snippet, a description of the code, some examples and several choices. Choose the correct option that completes the code snippet based on the question description.

            Rules:
            - Answer ONLY with the alphabet corresponding to the option (A, B, C, etc.)
            - No explanations, comments, or additional text
            - Use the provided examples to understand the pattern<|im_end|>
            <|im_start|>user
            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Examples
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?<|im_end|>
            <|im_start|>assistant
        """)
        return prompt


class QwenThinkingMCQInconsistencyPromptTemplate(PromptTemplate):
    """Qwen-3 specific MCQ inconsistency prompt templates with thinking mode enabled."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            You are given a code snippet, a description of the code and several choices. Think through the problem step by step, then choose the correct option that completes the code snippet.

            Rules:
            - Think through the problem in <think></think> tags
            - Answer ONLY with the alphabet corresponding to the option (A, B, C, etc.) after thinking
            - No explanations in the final answer<|im_end|>
            <|im_start|>user
            {qn_desc}
                        
            # Code Snippet
            {task}
            
            # Choices
            {choices}
                                
            Which option correctly completes the code snippet?<|im_end|>
            <|im_start|>assistant
            <think>
        """)
        return prompt
        
    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            You are given a code snippet, a description of the code, an example and several choices. Think through the problem step by step, then choose the correct option that completes the code snippet.

            Rules:
            - Think through the problem in <think></think> tags
            - Answer ONLY with the alphabet corresponding to the option (A, B, C, etc.) after thinking
            - Use the provided example to understand the pattern<|im_end|>
            <|im_start|>user
            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Example
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?<|im_end|>
            <|im_start|>assistant
            <think>
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|im_start|>system
            You are given a code snippet, a description of the code, some examples and several choices. Think through the problem step by step, then choose the correct option that completes the code snippet.

            Rules:
            - Think through the problem in <think></think> tags
            - Answer ONLY with the alphabet corresponding to the option (A, B, C, etc.) after thinking
            - Use the provided examples to understand the pattern<|im_end|>
            <|im_start|>user
            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Examples
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?<|im_end|>
            <|im_start|>assistant
            <think>
        """)
        return prompt


if __name__ == "__main__":
    pass