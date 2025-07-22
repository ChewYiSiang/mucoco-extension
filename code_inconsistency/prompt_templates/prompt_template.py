import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate

class CodeInconsistencyPromptTemplate(PromptTemplate):
    class OutputPrediction:
        def zero_shot_prompt() -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Expected Output: 
            """)
            return prompt
        
        def one_shot_prompt() -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, the input and a single example. You may use the example to determine the expected output. Return the expected output in your answer.
                
                # Your answer should only contain the expected output with no additional information. 
                                    
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Example
                {example}
                                    
                # Expected Output: 
            """)
            return prompt
        
        def few_shot_prompt() -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, the input and a few examples. You may use the examples to determine the expected output. Return the expected output in your answer. 
                
                # Your answer should only contain the expected output with no additional information. 
                                    
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Examples
                {example}
                                    
                # Expected Output: 
            """)
            return prompt

    class InputPrediction:
        def zero_shot_prompt() -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code and the output. Return the expected input in your answer that would produce this output. 
                # Your answer should only contain the expected input with no additional information and explanations. 
                                     
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Output
                {test_output}
                                    
                # Expected Input: 
            """)
            return prompt
        
        def one_shot_prompt() -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, an example and the output. You may use the example to determine the expected input. Return the expected input in your answer that would produce this output. 
                # Your answer should only contain the expected input with no additional information and explanations. 
                                    
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Output
                {test_output}
                                    
                # Example
                {example}
                                    
                # Expected Output: 
            """)
            return prompt
        
        def few_shot_prompt() -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, a few examples and the output. You may use the examples to determine the expected input. Return the expected input in your answer that would produce this output. 
                # Your answer should only contain the expected input with no additional information and explanations. 
                                    
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Output
                {test_output}
                                    
                # Examples
                {example}
                                    
                # Expected Output: 
            """)
            return prompt




if __name__ == "__main__":
    pass