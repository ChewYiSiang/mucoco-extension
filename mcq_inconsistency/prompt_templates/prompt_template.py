import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class MCQInconsistencyPromptTemplate(PromptTemplate):
    
    @staticmethod
    def return_model_appropriate_prompt(prompt_type: str, model_name: str = None, thinking_mode: bool = False):
        """Return the appropriate prompt template based on model type."""
        # Check if it's a Qwen model
        if model_name and ('qwen' in model_name.lower() or 'Qwen' in model_name):
            return MCQInconsistencyPromptTemplate.return_appropriate_qwen_prompt(prompt_type, thinking_mode)
        else:
            # Use generic templates for other models
            return MCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_qwen_prompt(prompt_type: str, thinking_mode: bool = False):
        """Return Qwen-specific prompts with ChatML format."""
        from mcq_inconsistency.prompt_templates.qwen_prompt_template import QwenMCQInconsistencyPromptTemplate, QwenThinkingMCQInconsistencyPromptTemplate
        
        if thinking_mode:
            return QwenThinkingMCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
        else:
            return QwenMCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C, etc. Do not give any additional details.
            {qn_desc}
                        
            # Code Snippet
            {task}
            
            # Choices
            {choices}
                                
            # Your answer
        """)
        return prompt
        
    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code, an example and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C etc. You may use the example to help answer the question. Do not give any additional details.
            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Example
            {example}
            
            # Choices
            {choices}
                                 
            # Your answer:
            
        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code, some examples and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C etc. You may use the examples to help answer the question. Do not give any additional details.
            {qn_desc}
                        
            # Code Snippet
            {task}
                                 
            # Examples
            {example}
            
            # Choices
            {choices}
                                 
            # Your answer:
            
        """)
        return prompt

if __name__ == "__main__":
    pass