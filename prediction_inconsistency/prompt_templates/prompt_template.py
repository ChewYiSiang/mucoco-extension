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
    
    @staticmethod
    def structure_few_shot_examples(test_cases: dict) -> str:
        """Structure examples for few shot prompts."""
        examples = []
        for input_case, expected_output in test_cases.items():
            examples.append(f"Input: {input_case}\nOutput: {expected_output}")
        return "\n\n".join(examples)
    
    @staticmethod
    def structure_one_shot_example(test_case: dict) -> str:
        """Structure example for one shot prompts."""
        for input_case, expected_output in test_case.items():
            return f"Input: {input_case}\nOutput: {expected_output}"
        return ""
    
    @staticmethod
    def return_appropriate_llama_prompt(task_type: str, prompt_type: str):
        """Return Llama-specific prompts with proper chat template format."""
        if task_type == OutputPrediction.NAME:
            return LlamaPredictionInconsistencyPromptTemplate.OutputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        elif task_type == InputPrediction.NAME:
            return LlamaPredictionInconsistencyPromptTemplate.InputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        else:
            raise ValueError(f"{task_type} is an invalid task type. Only {InputPrediction.NAME} and {OutputPrediction.NAME} are valid.")
    
    @staticmethod
    def return_model_appropriate_prompt(task_type: str, prompt_type: str, model_name: str = None):
        """Return the appropriate prompt template based on model type."""
        # Check if it's a Llama model
        if model_name and ('llama' in model_name.lower() or 'Llama' in model_name):
            return PredictionInconsistencyPromptTemplate.return_appropriate_llama_prompt(task_type, prompt_type)
        else:
            # Use generic templates for other models (Mistral, GPT, etc.)
            return PredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
        
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 
                                     
                # The output should be in the expected format. For example, given max([10,1]), your answer should be 10 and not "10".
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
                # The output should be in the expected format. For example, given max([10,1]), your answer should be 10 and not "10".

                                    
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
                # The output should be in the expected format. For example, given max([10,1]), your answer should be 10 and not "10".

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


class LlamaPredictionInconsistencyPromptTemplate:
    """Llama-specific prompt templates using official Llama 3.1 chat template format."""
    
    @staticmethod
    def return_appropriate_prompt(task_type: str, prompt_type: str):
        if task_type == OutputPrediction.NAME:
            return LlamaPredictionInconsistencyPromptTemplate.OutputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        elif task_type == InputPrediction.NAME:
            return LlamaPredictionInconsistencyPromptTemplate.InputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        else:
            raise ValueError(f"{task_type} is an invalid task type. Only {InputPrediction.NAME} and {OutputPrediction.NAME} are valid.")
    
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - For lists/tuples, use exact Python syntax: [(1, 2), (3, 4)]
                - For strings, include quotes if they would be in the output
                - For booleans, return True or False
                - For numbers, return the exact numeric value<|eot_id|><|start_header_id|>user<|end_header_id|>

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                What is the expected output when this code runs with the given input?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - Use the provided example to understand the expected format<|eot_id|><|start_header_id|>user<|end_header_id|>

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Example
                {example}

                What is the expected output when this code runs with the given input?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - Use the provided examples to understand the expected format<|eot_id|><|start_header_id|>user<|end_header_id|>

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Examples
                {example}

                What is the expected output when this code runs with the given input?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """)
            return prompt
    
    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"
                - No explanations, comments, or additional text
                - True = the input could produce the output
                - False = the input could not produce the output<|eot_id|><|start_header_id|>user<|end_header_id|>

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"  
                - No explanations, comments, or additional text
                - Use the provided example to understand the task<|eot_id|><|start_header_id|>user<|end_header_id|>

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Example
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"  
                - No explanations, comments, or additional text
                - Use the provided example to understand the task<|eot_id|><|start_header_id|>user<|end_header_id|>

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Examples
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """)
            return prompt


if __name__ == "__main__":
    pass