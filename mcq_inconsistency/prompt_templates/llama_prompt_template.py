import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class LlamaMCQInconsistencyPromptTemplate(PromptTemplate):
    """Llama-specific MCQ inconsistency prompt templates using official Llama 3.1 chat template format."""
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are given a code snippet, a description of the code and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C, etc.

            Rules:
            - Return ONLY the single letter (A, B, C, etc.)
            - No periods, punctuation, or extra formatting
            - No explanations, comments, or additional text
            - Do not include type information or tuple representations<|eot_id|><|start_header_id|>user<|end_header_id|>

            {qn_desc}
                        
            # Code Snippet
            {task}
            
            # Choices
            {choices}
                                
            Which option correctly completes the code snippet?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """)
        return prompt
        
    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are given a code snippet, a description of the code, an example and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C etc. You may use the example to help answer the question.

            Rules:
            - Return ONLY the single letter (A, B, C, etc.)
            - No periods, punctuation, or extra formatting
            - No explanations, comments, or additional text
            - Do not include type information or tuple representations<|eot_id|><|start_header_id|>user<|end_header_id|>

            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Example
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are given a code snippet, a description of the code, some examples and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C etc. You may use the examples to help answer the question.

            Rules:
            - Return ONLY the single letter (A, B, C, etc.)
            - No periods, punctuation, or extra formatting
            - No explanations, comments, or additional text
            - Do not include type information or tuple representations<|eot_id|><|start_header_id|>user<|end_header_id|>

            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Examples
            {example}
            
            # Choices
            {choices}
                                 
            Which option correctly completes the code snippet?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """)
        return prompt


if __name__ == "__main__":
    pass