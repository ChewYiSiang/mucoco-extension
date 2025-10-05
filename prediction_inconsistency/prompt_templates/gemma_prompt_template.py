import textwrap 
from code_generation.prompt_templates.prompt_template import PromptTemplate
from utility.constants import OutputPrediction, InputPrediction

class GemmaPredictionInconsistencyPromptTemplate:
    """Gemma-specific prompt templates using Gemma chat template format with <start_of_turn> tokens."""
    
    @staticmethod
    def return_appropriate_prompt(task_type: str, prompt_type: str):
        if task_type == OutputPrediction.NAME:
            return GemmaPredictionInconsistencyPromptTemplate.OutputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        elif task_type == InputPrediction.NAME:
            return GemmaPredictionInconsistencyPromptTemplate.InputPrediction().return_appropriate_prompt(prompt_type=prompt_type)
        else:
            raise ValueError(f"{task_type} is an invalid task type. Only {InputPrediction.NAME} and {OutputPrediction.NAME} are valid.")
    
    class OutputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <start_of_turn>user
                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information.

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - For lists/tuples, use exact Python syntax: [(1, 2), (3, 4)]
                - For strings, include quotes if they would be in the output
                - For booleans, return True or False
                - For numbers, return the exact numeric value

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                What is the expected output when this code runs with the given input?<end_of_turn>
                <start_of_turn>model
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <start_of_turn>user
                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information.

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - Use the provided example to understand the expected format

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Example
                {example}

                What is the expected output when this code runs with the given input?<end_of_turn>
                <start_of_turn>model
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <start_of_turn>user
                You are given a code snippet, a description of the code and the input. Return the expected output in your answer. Your answer should only contain the expected output with no additional information.

                Rules:
                - Return ONLY the expected output value
                - No explanations, comments, or additional text
                - Format the output exactly as Python would print it
                - Use the provided examples to understand the expected format

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Input
                {test_input}

                # Examples
                {example}

                What is the expected output when this code runs with the given input?<end_of_turn>
                <start_of_turn>model
            """)
            return prompt
    
    class InputPrediction(PromptTemplate):
        def zero_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <start_of_turn>user
                You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"
                - No explanations, comments, or additional text
                - True = the input could produce the output
                - False = the input could not produce the output

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<end_of_turn>
                <start_of_turn>model
            """)
            return prompt
        
        def one_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <start_of_turn>user
                You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"  
                - No explanations, comments, or additional text
                - Use the provided example to understand the task

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Example
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<end_of_turn>
                <start_of_turn>model
            """)
            return prompt
        
        def few_shot_prompt(self) -> str:
            prompt = textwrap.dedent("""
                <start_of_turn>user
                You are given a code snippet, a description of the code, an output and an input. Your task is to determine if running the program with the input could result in the output.

                Rules:
                - Answer ONLY with "True" or "False"  
                - No explanations, comments, or additional text
                - Use the provided examples to understand the task

                {qn_desc}

                # Code Snippet
                {full_sol}

                # Examples
                {example}

                # Output
                {test_output}

                # Input
                {test_input}

                Could running this code with the given input produce the given output?<end_of_turn>
                <start_of_turn>model
            """)
            return prompt


if __name__ == "__main__":
    pass