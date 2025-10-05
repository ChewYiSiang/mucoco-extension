import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class DeepSeekMCQInconsistencyPromptTemplate(PromptTemplate):
    """DeepSeek-specific MCQ inconsistency prompt templates using DeepSeek chat template format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <｜begin▁of▁sentence｜>User: You are given a code snippet, a description of the code and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C, etc. Do not give any additional details.

            {qn_desc}
                        
            # Code Snippet
            {task}
            
            # Choices
            {choices}
                                
            Which option correctly completes the code snippet?

            Assistant: """)
        return prompt
        
    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <｜begin▁of▁sentence｜>User: You are given a code snippet, a description of the code, an example and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C etc. You may use the example to help answer the question. Do not give any additional details.

            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Example
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?

            Assistant: """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <｜begin▁of▁sentence｜>User: You are given a code snippet, a description of the code, some examples and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C etc. You may use the examples to help answer the question. Do not give any additional details.

            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Examples
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?

            Assistant: """)
        return prompt


if __name__ == "__main__":
    pass