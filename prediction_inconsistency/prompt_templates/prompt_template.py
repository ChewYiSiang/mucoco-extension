import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate
from utility.constants import OutputPrediction, InputPrediction

class PredictionInconsistencyPromptTemplate:
    def return_appropriate_prompt(task_type: str, prompt_type: str):
        if task_type == OutputPrediction.NAME:
            return PredictionInconsistencyPromptTemplate.OutputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        elif task_type == InputPrediction.NAME:
            return PredictionInconsistencyPromptTemplate.InputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        else:
            raise ValueError(f"{task_type} is an invalid task type. Only {InputPrediction.NAME} and {OutputPrediction.NAME} are valid.")
        
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Expected Output:                      
                ### Your answer
            """)
            return prompt
            
        def one_shot_prompt(self) -> str:
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
                ### Your answer
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
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
                ### Your answer
            """)
            return prompt

    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output. 
                # Your answer should either be "True" or "False". Do not provide any additional information and explanations. 
                                     
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Output
                {test_output}
                                     
                # Input
                {test_input}
                                    
                # Your Answer
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output. You are also provided an example, which you may use to answer the question.
                # Your answer should either be "True" or "False". Do not provide any additional information and explanations. 
                                                   
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                                     
                # Examples
                {example}
                
                # Output
                {test_output}
                                     
                # Input
                {test_input}
                                    
                # Your Answer
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output. You are also provided with some examples, which you may use to answer the question.
                # Your answer should either be "True" or "False". Do not provide any additional information and explanations. 
                                    
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                                     
                # Examples
                {example}
                                     
                # Output
                {test_output}
                                     
                # Input
                {test_input}
                                    
                # Your Answer
            """)
            return prompt


if __name__ == "__main__":
    pass