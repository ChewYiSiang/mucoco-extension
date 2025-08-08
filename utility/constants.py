from code_generation.prompt_templates.prompt_template import OpenEndedPromptTemplate, LlamaOpenEndedPromptTemplate
from code_inconsistency.prompt_templates.prompt_template import CodeInconsistencyPromptTemplate
from typing import Callable

class PromptTypes:
    ZERO_SHOT = 'zero_shot'
    ONE_SHOT = 'one_shot'
    FEW_SHOT = 'few_shot'

class SyntacticMutations:
    FOR2WHILE = 'for2while'
    FOR2ENUMERATE = 'for2enumerate'

class LexicalMutations:
    RANDOM = 'random'
    SEQUENTIAL = 'sequential'

class TaskTypes:
    OUTPUT_PREDICTION = 'output_prediction'
    INPUT_PREDICTION = 'input_prediction'

class PromptConfig:
    def __init__(self, prompt_helper: Callable, example_helper: Callable | None):
        self.prompt_helper = prompt_helper
        self.example_helper = example_helper

CODE_INCONSISTENCY_PROMPT_CONFIG ={
    "general" : {
        "zero_shot" : PromptConfig(
            prompt_helper = CodeInconsistencyPromptTemplate.OutputPrediction.zero_shot_prompt, 
            example_helper = None
            ),
        "one_shot": PromptConfig(
            prompt_helper = CodeInconsistencyPromptTemplate.OutputPrediction.one_shot_prompt, 
            example_helper = CodeInconsistencyPromptTemplate.structure_one_shot_example
            ),
        "few_shot" : PromptConfig(
            prompt_helper = CodeInconsistencyPromptTemplate.OutputPrediction.few_shot_prompt,
            example_helper = CodeInconsistencyPromptTemplate.structure_few_shot_examples
        ),
    },
    "llama": {
        "zero_shot" : {},
        "one_shot": {},
        "few_shot" : {},
        
    }
}