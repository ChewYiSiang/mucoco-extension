import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class MCQInconsistencyPromptTemplate(PromptTemplate):
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