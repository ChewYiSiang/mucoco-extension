import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class MCQInconsistencyPromptTemplate(PromptTemplate):
    
    @staticmethod
    def return_model_appropriate_prompt(prompt_type: str, model_name: str = None, thinking_mode: bool = False):
        """Return the appropriate prompt template based on model type."""
        # Check if it's a Qwen model
        if model_name and ('qwen' in model_name.lower() or 'Qwen' in model_name):
            return MCQInconsistencyPromptTemplate.return_appropriate_qwen_prompt(prompt_type, thinking_mode)
        # Check if it's a Gemma model
        elif model_name and ('gemma' in model_name.lower() or 'Gemma' in model_name):
            return MCQInconsistencyPromptTemplate.return_appropriate_gemma_prompt(prompt_type)
        # Check if it's a DeepSeek model
        elif model_name and ('deepseek' in model_name.lower() or 'DeepSeek' in model_name):
            return MCQInconsistencyPromptTemplate.return_appropriate_deepseek_prompt(prompt_type)
        # Check if it's a Mistral model
        elif model_name and ('mistral' in model_name.lower() or 'Mistral' in model_name):
            return MCQInconsistencyPromptTemplate.return_appropriate_mistral_prompt(prompt_type)
        # Check if it's a Llama model
        elif model_name and ('llama' in model_name.lower() or 'Llama' in model_name):
            return MCQInconsistencyPromptTemplate.return_appropriate_llama_prompt(prompt_type)
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
    
    @staticmethod
    def return_appropriate_gemma_prompt(prompt_type: str):
        """Return Gemma-specific prompts with Gemma chat template format."""
        from mcq_inconsistency.prompt_templates.gemma_prompt_template import GemmaMCQInconsistencyPromptTemplate
        
        return GemmaMCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_deepseek_prompt(prompt_type: str):
        """Return DeepSeek-specific prompts with DeepSeek chat template format."""
        from mcq_inconsistency.prompt_templates.deepseek_prompt_template import DeepSeekMCQInconsistencyPromptTemplate
        
        return DeepSeekMCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_mistral_prompt(prompt_type: str):
        """Return Mistral-specific prompts with Mistral chat template format."""
        from mcq_inconsistency.prompt_templates.mistral_prompt_template import MistralMCQInconsistencyPromptTemplate
        
        return MistralMCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_llama_prompt(prompt_type: str):
        """Return Llama-specific prompts with Llama 3.1 chat template format."""
        from mcq_inconsistency.prompt_templates.llama_prompt_template import LlamaMCQInconsistencyPromptTemplate
        
        return LlamaMCQInconsistencyPromptTemplate().return_appropriate_prompt(prompt_type)
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            # You are given a code snippet, a description of the code and several choices. Choose the correct option that completes the code snippet based on the question description. Your answer should only be the alphabet corresponding to the option i.e.: A, B, C, etc. Do not give any additional details.
            # Do not return any reasoning in your final answer.

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
            # Do not return any reasoning in your final answer.
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
            # Do not return any reasoning steps. 
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
    

class ReasoningMCQInconsistencyPromptTemplate(PromptTemplate):
    def zero_shot_prompt(self):
        prompt = textwrap.dedent("""
            # Return the correct option in this Multiple Choice Question that completes the program according to the task description.
            # You must ahere to the following instructions:
            # - Use the task description to make your choice.
            # - Only return the letter corresponding to your choice.
            # - Do not include any intermediate steps or any reasoning steps in your answer.
            
            ### Task Description
            {qn_desc}
                                    
            ### Code Snippet
            {task}
            
            ### Choices
            {choices}
                                
            ### Your Answer:
        """)

        return prompt


    def one_shot_prompt(self):
        prompt = textwrap.dedent("""
            # Return the correct option in this Multiple Choice Question that completes the program according to the task description.
            # You must ahere to the following instructions:
            # - Use the task description and example to make your choice.
            # - Only return the letter corresponding to your choice.
            # - Do not include any intermediate steps or any reasoning steps in your answer.
            
            ### Task Description
            {qn_desc}
                                    
            ### Code Snippet
            {task}
                                 
            ### Example
            {example}
            
            ### Choices
            {choices}
                                
            ### Your Answer:
        """)

        return prompt
    
    def few_shot_prompt(self):
        prompt = textwrap.dedent("""
            # Return the correct option in this Multiple Choice Question that completes the program according to the task description.
            # You must ahere to the following instructions:
            # - Use the task description and examples to make your choice.
            # - Only return the letter corresponding to your choice.
            # - Do not include any intermediate steps or any reasoning steps in your answer.
            
            ### Task Description
            {qn_desc}
                                    
            ### Code Snippet
            {task}
                                 
            ### Examples
            {example}
            
            ### Choices
            {choices}
                                
            ### Your Answer:
        """)
                
        return prompt
        

if __name__ == "__main__":
    pass