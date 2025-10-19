from abc import ABC, abstractmethod
import textwrap 
from typing import List, Dict, Callable
import random
from utility.constants import PromptTypes

class PromptTemplate(ABC):
    def return_appropriate_prompt(self, prompt_type: str) -> Callable:
        valid_prompt_types = [getattr(PromptTypes, prompt) for prompt in dir(PromptTypes)]
        if prompt_type not in valid_prompt_types:
            raise ValueError(f"An invalid prompt type was used. Only {valid_prompt_types} are valid.")
        
        if prompt_type == PromptTypes.ZERO_SHOT:
            return self.zero_shot_prompt
        elif prompt_type == PromptTypes.ONE_SHOT:
            return self.one_shot_prompt
        elif prompt_type == PromptTypes.FEW_SHOT:
            return self.few_shot_prompt

    @staticmethod
    def structure_few_shot_examples(test_cases: Dict[str, str]) -> str:
        """
        This function is used to format examples in a dictionary into a standardised doctest format to be included in a few shot prompt template. This is used to structure the examples for few shot prompts.

        Args: 
            test_cases (Dict[str, str]): a dictionary containing examples with inputs and their respective expected outputs. 

                The dictionary keys are function calls with inputs. The value for the dicionary keys is the expected output from executing the function and input.

        Returns:
            str: a string representing the structured few shot examples
        """
        return "\n".join(">>> " + test + "\n" + test_cases[test] for test in test_cases)
    
    @staticmethod
    def structure_one_shot_example(test_cases: Dict[str, str] | str) -> str:
        """
        This function is used to format examples in a dictionary into a standardised doctest format to be included in a one shot prompt template. This is used to structure the examples used for one shot prompts.

        Args: 
            test_cases (Dict[str, str]): a dictionary containing examples with inputs and their respective expected outputs. 

                The dictionary keys are function calls with inputs. The value for the dicionary keys is the expected output from executing the function and input.

        Returns:
            str: a string representing the structured one shot examples
        """

        if isinstance(test_cases, Dict):
            random_example = random.choice(list(test_cases.keys()))
            return ">>> " + random_example + "\n" + test_cases[random_example]
        else:
            return test_cases
        
    @abstractmethod
    def zero_shot_prompt(self) -> str:
        pass

    @abstractmethod
    def one_shot_prompt(self) -> str:
        pass

    @abstractmethod
    def few_shot_prompt(self) -> str:
        pass

class MCQPromptTemplate(PromptTemplate):
    
    @staticmethod
    def return_model_appropriate_prompt(prompt_type: str, model_name: str = None, thinking_mode: bool = False):
        """Return the appropriate prompt template based on model type."""
        # Check if it's a Qwen model
        if model_name and ('qwen' in model_name.lower()):
            return MCQPromptTemplate.return_appropriate_qwen_prompt(prompt_type, thinking_mode)
        # Check if it's a Gemma model
        elif model_name and ('gemma' in model_name.lower()):
            return MCQPromptTemplate.return_appropriate_gemma_prompt(prompt_type)
        # Check if it's a DeepSeek model
        elif model_name and ('deepseek' in model_name.lower()):
            return MCQPromptTemplate.return_appropriate_deepseek_prompt(prompt_type)
        # Check if it's a Mistral Codestral model
        elif model_name and ('codestral' in model_name.lower()):
            return MCQPromptTemplate.return_appropriate_mistral_prompt(prompt_type)
        else:
            # Use generic templates for other models
            return MCQPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_qwen_prompt(prompt_type: str, thinking_mode: bool = False):
        """Return Qwen-specific prompts with ChatML format."""
        from code_generation.prompt_templates.qwen_prompt_template import QwenMCQPromptTemplate
        
        return QwenMCQPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_gemma_prompt(prompt_type: str):
        """Return Gemma-specific prompts with Gemma chat template format."""
        from code_generation.prompt_templates.gemma_prompt_template import GemmaMCQPromptTemplate
        
        return GemmaMCQPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_deepseek_prompt(prompt_type: str):
        """Return DeepSeek-specific prompts with DeepSeek chat template format."""
        from code_generation.prompt_templates.deepseek_prompt_template import DeepSeekMCQPromptTemplate
        
        return DeepSeekMCQPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_mistral_prompt(prompt_type: str):
        """Return Mistral-specific prompts with Mistral chat template format."""
        from code_generation.prompt_templates.mistral_prompt_template import MistralMCQPromptTemplate
        
        return MistralMCQPromptTemplate().return_appropriate_prompt(prompt_type)
    
    def zero_shot_prompt() -> str:
        prompt = textwrap.dedent("""
            Using this code snipper, answer the MCQ question below. 
            {question}

            Test case: {test_case}
                                 
            If this code above is run with the test case, what will the answer be? Only return your final answer below as A, B, C, D, or E.

            Options
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            E) {E}

            Your answer: 
        """)
        return prompt

class OpenEndedPromptTemplate(PromptTemplate):
    
    @staticmethod
    def return_model_appropriate_prompt(prompt_type: str, model_name: str = None, thinking_mode: bool = False):
        """Return the appropriate prompt template based on model type."""
        # Check if it's a Llama model
        if model_name and ('llama' in model_name.lower() or 'Llama' in model_name):
            return LlamaOpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
        # Check if it's a Qwen model
        elif model_name and ('qwen' in model_name.lower() or 'Qwen' in model_name):
            return OpenEndedPromptTemplate.return_appropriate_qwen_prompt(prompt_type, thinking_mode)
        # Check if it's a Gemma model
        elif model_name and ('gemma' in model_name.lower() or 'Gemma' in model_name):
            return OpenEndedPromptTemplate.return_appropriate_gemma_prompt(prompt_type)
        # Check if it's a DeepSeek model
        elif model_name and ('deepseek' in model_name.lower() or 'DeepSeek' in model_name):
            return OpenEndedPromptTemplate.return_appropriate_deepseek_prompt(prompt_type)
        # Check if it's a Mistral codestral model
        elif model_name and ('codestral' in model_name.lower()):
            return OpenEndedPromptTemplate.return_appropriate_mistral_prompt(prompt_type)
        else:
            # Use generic templates for other models
            return OpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_qwen_prompt(prompt_type: str, thinking_mode: bool = False):
        """Return Qwen-specific prompts with ChatML format."""
        from code_generation.prompt_templates.qwen_prompt_template import QwenOpenEndedPromptTemplate, QwenThinkingOpenEndedPromptTemplate
        
        if thinking_mode:
            return QwenThinkingOpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
        else:
            return QwenOpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_gemma_prompt(prompt_type: str):
        """Return Gemma-specific prompts with Gemma chat template format."""
        from code_generation.prompt_templates.gemma_prompt_template import GemmaOpenEndedPromptTemplate
        
        return GemmaOpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_deepseek_prompt(prompt_type: str):
        """Return DeepSeek-specific prompts with DeepSeek chat template format."""
        from code_generation.prompt_templates.deepseek_prompt_template import DeepSeekOpenEndedPromptTemplate
        
        return DeepSeekOpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
    
    @staticmethod
    def return_appropriate_mistral_prompt(prompt_type: str):
        """Return Mistral-specific prompts with Mistral chat template format."""
        from code_generation.prompt_templates.mistral_prompt_template import MistralOpenEndedPromptTemplate
        
        return MistralOpenEndedPromptTemplate().return_appropriate_prompt(prompt_type)
    
    def zero_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
        # Complete the given code snippet using the description below. 
        # Complete the function body and do not add any explanations or extra text. 
        # Return the given code snippet in your answer. 
        # Preserve indentation.

        ### Example:

        # Task
        Return a function that returns the largest integer in a list.

        # Code Snippet
        def find_max(nums):

        # Your Answer
        def find_max(nums):
            return max(nums)

        ### Task:
        {task}

        ### Code Snippet:
        {code}

        # Your Answer:
        """)
        return prompt

    def one_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            # Complete the code for the following function given it's description. Only complete the code function and do not add any other details. You may use the given example to write your code. Return your answer as a complete function, including any provided code. 
            {task}
                                 
            # Example:
            {example}
                                 
            # Your answer: 
            {code}

        """)
        return prompt
    
    def few_shot_prompt(self) -> str:
        prompt = textwrap.dedent("""
            # Complete the code for the following function given it's description. You may use the given examples to write your code. Return your answer as a complete function.
            {task}

            # Examples:
            {example}
                                 
            # Your answer: 
            {code}
        """)
        return prompt

class LlamaOpenEndedPromptTemplate(OpenEndedPromptTemplate):
    def zero_shot_prompt(self):
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a Python coding assistant. Complete the given function implementation. Only provide the complete function code including any necessary imports without any explanations, comments, test cases, or additional text. Do not include any tokens like <|end_of_text|> in your response.
            {task}<|eot_id|>
                                 
            <|start_header_id|>user<|end_header_id|>
            Complete this function:
            
            {code}<|eot_id|>
                                 
            <|start_header_id|>assistant<|end_header_id|>
        """)
        return prompt

    def one_shot_prompt(self):
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a Python coding assistant. Complete the given function implementation. Only provide the complete function code including any necessary imports without any explanations, comments, test cases, or additional text. Do not include any tokens like <|end_of_text|> in your response.
            {task}<|eot_id|>
                                 
            <|start_header_id|>user<|end_header_id|>
            Complete this function:
            
            {code}
            
            Example usage:
            {example}<|eot_id|>
                                 
            <|start_header_id|>assistant<|end_header_id|>
        """)
        return prompt

    def few_shot_prompt(self):
        prompt = textwrap.dedent("""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            # Complete the code for the following function given it's description. You may use the given examples to write your code. Return your answer as a complete function including any necessary imports.
            {task}<|eot_id|>
                                 
            <|start_header_id|>user<|end_header_id|>
            # Code Snippet:
            {code}
                                 
            # Examples:
            {example}<|eot_id|>
                                 
            # Your answer:
            <|start_header_id|>assistant<|end_header_id|>
        """)
        return prompt
    
class ReasoningOpenEndedPromptTemplate(OpenEndedPromptTemplate):
    def zero_shot_prompt(self):
        prompt = textwrap.dedent("""
            # Complete the given code function using the task description. 
            # You must ahere to the following instructions:
            # - Complete the given code snippet exactly as provided.
            # - Include the code snippet in your answer.
            # - Do not include any intermediate steps or any reasoning steps in your answer.
            # - Your final answer should be valid Python code ready to be run.
                                 
            ### Example:

            # Task
            Return a function that returns the largest integer in a list.

            # Code Snippet
            def find_max(nums):

            # Your Answer
            def find_max(nums):
                return max(nums)

            ### Task Description:
            {task}

            ## Examples:
            {example}
                                 
            ### Code Snippet:
            {code}

            ### Your Answer:
        """)
        return prompt
    
    def one_shot_prompt(self):
        prompt = textwrap.dedent("""
            # Complete the given code function using the task description and example. 
            # You must ahere to the following instructions:
            # - Complete the given code snippet exactly as provided.
            # - Include the code snippet in your answer.
            # - Do not include any intermediate steps or any reasoning steps in your answer.
            # - Your final answer should be valid Python code ready to be run.
                                 
            ### Example:

            # Task
            Return a function that returns the largest integer in a list.

            # Code Snippet
            def find_max(nums):

            # Your Answer
            def find_max(nums):
                return max(nums)

            ### Task Description:
            {task}

            ### Examples:
            {example}
                                 
            ### Code Snippet:
            {code}

            ### Your Answer:
        """)
        return prompt
    
    def few_shot_prompt(self):
        prompt = textwrap.dedent("""
            # Complete the given code function using the task description and examples. 
            # You must ahere to the following instructions:
            # - Complete the given code snippet exactly as provided.
            # - Include the code snippet in your answer.
            # - Do not include any intermediate steps or any reasoning steps in your answer.
            # - Your final answer should be valid Python code ready to be run.
                                 
            ### Example:

            # Task
            Return a function that returns the largest integer in a list.

            # Code Snippet
            def find_max(nums):

            # Your Answer
            def find_max(nums):
                return max(nums)

            ### Task Description:
            {task}

            ### Examples:
            {example}
                                 
            ### Code Snippet:
            {code}

            ### Your Answer:
        """)
        return prompt


if __name__ == "__main__":
    x = {
        "func1": "abc",
        "func2": "def"
    }
    tc = OpenEndedPromptTemplate.structure_few_shot_examples(x)
    print(tc)