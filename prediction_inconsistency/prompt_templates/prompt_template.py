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
    def return_appropriate_qwen_prompt(task_type: str, prompt_type: str, thinking_mode: bool = False):
        """Return Qwen-specific prompts with ChatML format."""
        from prediction_inconsistency.prompt_templates.qwen_prompt_template import QwenPredictionInconsistencyPromptTemplate, QwenThinkingPredictionInconsistencyPromptTemplate
        
        if thinking_mode:
            return QwenThinkingPredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
        else:
            return QwenPredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
    
    @staticmethod
    def return_appropriate_gemma_prompt(task_type: str, prompt_type: str):
        """Return Gemma-specific prompts with Gemma chat template format."""
        from prediction_inconsistency.prompt_templates.gemma_prompt_template import GemmaPredictionInconsistencyPromptTemplate
        
        return GemmaPredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
    
    @staticmethod
    def return_appropriate_deepseek_prompt(task_type: str, prompt_type: str):
        """Return DeepSeek-specific prompts with DeepSeek chat template format."""
        from prediction_inconsistency.prompt_templates.deepseek_prompt_template import DeepSeekPredictionInconsistencyPromptTemplate
        
        return DeepSeekPredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
    
    @staticmethod
    def return_appropriate_mistral_prompt(task_type: str, prompt_type: str):
        """Return Mistral-specific prompts with Mistral chat template format."""
        from prediction_inconsistency.prompt_templates.mistral_prompt_template import MistralPredictionInconsistencyPromptTemplate
        
        return MistralPredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
    
    @staticmethod
    def return_model_appropriate_prompt(task_type: str, prompt_type: str, model_name: str = None, thinking_mode: bool = False):
        """Return the appropriate prompt template based on model type."""
        # Check if it's a Llama model
        if model_name and ('llama' in model_name.lower() or 'Llama' in model_name):
            return PredictionInconsistencyPromptTemplate.return_appropriate_llama_prompt(task_type, prompt_type)
        # Check if it's a Qwen model
        elif model_name and ('qwen' in model_name.lower() or 'Qwen' in model_name):
            return PredictionInconsistencyPromptTemplate.return_appropriate_qwen_prompt(task_type, prompt_type, thinking_mode)
        # Check if it's a Gemma model
        elif model_name and ('gemma' in model_name.lower() or 'Gemma' in model_name):
            return PredictionInconsistencyPromptTemplate.return_appropriate_gemma_prompt(task_type, prompt_type)
        # Check if it's a DeepSeek model
        elif model_name and ('deepseek' in model_name.lower() or 'DeepSeek' in model_name):
            return PredictionInconsistencyPromptTemplate.return_appropriate_deepseek_prompt(task_type, prompt_type)
        # Check if it's a Mistral model
        elif model_name and ('mistral' in model_name.lower() or 'Mistral' in model_name):
            return PredictionInconsistencyPromptTemplate.return_appropriate_mistral_prompt(task_type, prompt_type)
        else:
            # Use generic templates for other models (GPT, etc.)
            return PredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)
        
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information. 
                                     
                # The output should be in the expected format. For example, given max([10,1]), your answer should be 10 and not "10".
                # Do not return any reasoning in your final answer.
                                     
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Your answer
            """)
            return prompt
            
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, the input and a single example. You may use the example to determine the expected output. Return the expected output in your answer.
                # Your answer should only contain the expected output with no additional information. 
                # The output should be in the expected format. For example, given max([10,1]), your answer should be 10 and not "10".
                # Do not return any reasoning in your final answer.
                                    
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Example
                {example}
                                    
                # Your answer
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, the input and a few examples. You may use the examples to determine the expected output. Return the expected output in your answer. 
                # Your answer should only contain the expected output with no additional information. 
                # The output should be in the expected format. For example, given max([10,1]), your answer should be 10 and not "10".
                # Do not return any reasoning in your final answer.
                                     
                {qn_desc}
                            
                # Code Snippet
                {full_sol}
                
                # Input
                {test_input}
                                    
                # Examples
                {example}
                                    
                # Your answer
            """)
            return prompt

    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output. 
                # Your answer should either be "True" or "False". Do not provide any additional information and explanations. 
                # Do not return any reasoning in your final answer.
                          
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
                # Do not return any reasoning in your final answer.
                                           
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
                # Do not return any reasoning in your final answer.
                      
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
        

class ReasoningPredictionInconsistencyPromptTemplate(PredictionInconsistencyPromptTemplate):
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # Your task is to return the program output expected from the code program using the program description and program input.
                # You must adhere to the following instructions:
                # - Use the task description, code program and input to determine the program output
                # - Your final answer should only contain the output in the expected format. For example, given code program max([10,1]), your answer should be an integer 10 and not a string "10".
                # - Do not include any intermediate steps or any reasoning steps in your final answer.

                ### Program Description          
                {qn_desc}
                            
                ### Code Program
                {full_sol}
                
                ### Program Input
                {test_input}
                                    
                ### Your answer
            """)
            return prompt
            
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # Your task is to return the program output expected from the code program using the program description, program input and example.
                # You must adhere to the following instructions:
                # - Use the task description, code program, input and example to determine the program output
                # - Your final answer should only contain the output in the expected format. For example, given code program max max([10,1]), your answer should be an integer 10 and not a string "10".
                # - Do not include any intermediate steps or any reasoning steps in your final answer.

                ### Program Description            
                {qn_desc}
                            
                ### Code Program
                {full_sol}
                
                ### Program Input
                {test_input}
                                    
                ### Example
                {example}
                                    
                ### Your answer
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # Your task is to return the program output expected from the code program using the program description, program input and examples.
                # You must adhere to the following instructions:
                # - Use the task description, code program, input and example to determine the program output
                # - Your final answer should only contain the output in the expected format. For example, given code program max max([10,1]), your answer should be an integer 10 and not a string "10".
                # - Do not include any intermediate steps or any reasoning steps in your final answer.

                ### Program Description            
                {qn_desc}
                            
                ### Code Program
                {full_sol}
                
                ### Program Input
                {test_input}
                                    
                ### Examples
                {example}
                                    
                ### Your answer
            """)
            return prompt
        
    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # Your task is to check whether running the given code with the provided input produces the specified output. 
                # You must adhere to the following instructions:
                # - Use the code, input, output, and description to decide.
                # - Your final answer should only be True or False.
                # - Return your final answer as a boolean value, not as a string.
                # - Do not include any intermediate steps or any reasoning steps in your final answer.

                ### Program Description          
                {qn_desc}
                            
                ### Code Snippet
                {full_sol}
                
                ### Output
                {test_output}
                                     
                ### Input
                {test_input}
                                    
                ### Your Answer
            """)
            return prompt
            
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # Your task is to check whether running the given code with the provided input produces the specified output. 
                # You must adhere to the following instructions:
                # - Use the code, input, output, description and example to decide.
                # - Your final answer should only be True or False.
                # - Return your final answer as a boolean value, not as a string.
                # - Do not include any intermediate steps or any reasoning steps in your final answer.
                                     
                ### Program Description          
                {qn_desc}
                            
                ### Code Snippet
                {full_sol}
                                     
                ### Examples
                {example}
                
                ### Output
                {test_output}
                                     
                ### Input
                {test_input}
                                    
                ### Your Answer
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                # Your task is to check whether running the given code with the provided input produces the specified output. 
                # You must adhere to the following instructions:
                # - Use the code, input, output, description and example to decide.
                # - Your final answer should only be True or False.
                # - Return your final answer as a boolean value, not as a string.
                # - Do not include any intermediate steps or any reasoning steps in your final answer.
                                     
                ### Program Description          
                {qn_desc}
                            
                ### Code Snippet
                {full_sol}
                                     
                ### Examples
                {example}
                
                ### Output
                {test_output}
                                     
                ### Input
                {test_input}
                                    
                ### Your Answer
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