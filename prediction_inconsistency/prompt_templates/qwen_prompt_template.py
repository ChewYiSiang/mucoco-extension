import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate
from utility.constants import OutputPrediction, InputPrediction

class QwenPredictionInconsistencyPromptTemplate:
    """Qwen-specific prompt templates using ChatML format with special tokens."""
    
    @staticmethod
    def return_appropriate_prompt(task_type: str, prompt_type: str):
        if task_type == OutputPrediction.NAME:
            return QwenPredictionInconsistencyPromptTemplate.OutputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        elif task_type == InputPrediction.NAME:
            return QwenPredictionInconsistencyPromptTemplate.InputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        else:
            raise ValueError(f"{task_type} is an invalid task type. Only {InputPrediction.NAME} and {OutputPrediction.NAME} are valid.")
    
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information.

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - For lists/tuples, use exact Python syntax: [(1, 2), (3, 4)]
                - For strings, include quotes if they would be in the output
                - For booleans, return True or False
                - For numbers, return the exact numeric value<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                What is the expected output when this code runs with the given input?<|im_end|>
                <|im_start|>assistant
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information.

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - Use the provided example to understand the expected format<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Example
                {example}

                What is the expected output when this code runs with the given input?<|im_end|>
                <|im_start|>assistant
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information.

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - Use the provided examples to understand the expected format<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Examples
                {example}

                What is the expected output when this code runs with the given input?<|im_end|>
                <|im_start|>assistant
            """)
            return prompt
    
    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"
                - No explanations, comments, or additional text
                - True = the input could produce the output
                - False = the input could not produce the output<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|im_end|>
                <|im_start|>assistant
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"  
                - No explanations, comments, or additional text
                - Use the provided example to understand the task<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Example
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|im_end|>
                <|im_start|>assistant
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"  
                - No explanations, comments, or additional text
                - Use the provided examples to understand the task<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Examples
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|im_end|>
                <|im_start|>assistant
            """)
            return prompt


class QwenThinkingPredictionInconsistencyPromptTemplate:
    """Qwen-3 specific prompt templates with thinking mode enabled."""
    
    @staticmethod
    def return_appropriate_prompt(task_type: str, prompt_type: str):
        if task_type == OutputPrediction.NAME:
            return QwenThinkingPredictionInconsistencyPromptTemplate.OutputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        elif task_type == InputPrediction.NAME:
            return QwenThinkingPredictionInconsistencyPromptTemplate.InputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        else:
            raise ValueError(f"{task_type} is an invalid task type. Only {InputPrediction.NAME} and {OutputPrediction.NAME} are valid.")
    
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code and the input. Think through the problem step by step, then return the expected output. Your final answer should only contain the expected output with no additional information.

                Rules:
                - Think through the problem in <think></think> tags
                - Return ONLY the expected output value after thinking
                - No explanations in the final answer
                - Format the output exactly as Python would print it<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                What is the expected output when this code runs with the given input?<|im_end|>
                <|im_start|>assistant
                <think>
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code and the input. Think through the problem step by step, then return the expected output. Your final answer should only contain the expected output with no additional information.

                Rules:
                - Think through the problem in <think></think> tags
                - Return ONLY the expected output value after thinking
                - Use the provided example to understand the expected format<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Example
                {example}

                What is the expected output when this code runs with the given input?<|im_end|>
                <|im_start|>assistant
                <think>
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code and the input. Think through the problem step by step, then return the expected output. Your final answer should only contain the expected output with no additional information.

                Rules:
                - Think through the problem in <think></think> tags
                - Return ONLY the expected output value after thinking
                - Use the provided examples to understand the expected format<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Examples
                {example}

                What is the expected output when this code runs with the given input?<|im_end|>
                <|im_start|>assistant
                <think>
            """)
            return prompt
    
    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code, an output and an input. Think through the problem step by step to determine if running the program with the input could result in the output.

                Rules:
                - Think through the problem in <think></think> tags
                - Answer ONLY with "True" or "False" after thinking
                - True = the input could produce the output
                - False = the input could not produce the output<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|im_end|>
                <|im_start|>assistant
                <think>
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code, an output and an input. Think through the problem step by step to determine if running the program with the input could result in the output.

                Rules:
                - Think through the problem in <think></think> tags
                - Answer ONLY with "True" or "False" after thinking
                - Use the provided example to understand the task<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Example
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|im_end|>
                <|im_start|>assistant
                <think>
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <|im_start|>system
                You are given a code snippet, a description of the code, an output and an input. Think through the problem step by step to determine if running the program with the input could result in the output.

                Rules:
                - Think through the problem in <think></think> tags
                - Answer ONLY with "True" or "False" after thinking
                - Use the provided examples to understand the task<|im_end|>
                <|im_start|>user
                {qn_desc}

                # Code Snippet
                {full_sol}

                # Examples
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<|im_end|>
                <|im_start|>assistant
                <think>
            """)
            return prompt


if __name__ == "__main__":
    pass