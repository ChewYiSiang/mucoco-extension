from prediction_inconsistency.prompt_templates.prompt_template import PredictionInconsistencyPromptTemplate
from typing import Callable

class PromptTypes:
    ZERO_SHOT = 'zero_shot'
    ONE_SHOT = 'one_shot'
    FEW_SHOT = 'few_shot'

class Mutations:
    class SyntacticMutations:
        FOR2WHILE = 'for2while'                                 # converting for loops to while loops
        FOR2ENUMERATE = 'for2enumerate'                         # converting for loops to include enumerate function

    class LexicalMutations:
        RANDOM = 'random'                                       # randomly mutate function and input variable names
        SEQUENTIAL = 'sequential'                               # mutate funciton and input variable names into a genetic naming convention
        LITERAL_FORMAT = "literal_format"                       # standardises strings from "hello" to 'hello' and vice versa

    class LogicalMutations:
        DEMORGAN = 'demorgan'                                   # applies demorgan transformation onto boolean statements
        BOOLEAN_LITERAL = "boolean_literal"                     # converts boolean literal representations: E.g.: True -> not False
        COMMUTATIVE_REORDER = "commutative_reorder"             # applied functionality preserving commutative operations

        CONSTANT_UNFOLD = "constant_unfold"                     # unfolds constant expression into either multiplication or addition statements
        CONSTANT_UNFOLD_ADD = "constant_unfold_add"             # unfolds constant expression into addition statements
        CONSTANT_UNFOLD_MULT = "constant_unfold_mult"           # unfolds constant expression into multiplication statements

SyntacticMutations = Mutations.SyntacticMutations
LexicalMutations = Mutations.LexicalMutations
LogicalMutations = Mutations.LogicalMutations

class Benchmarks:
    class HumanEval:
        NAME = "HumanEval"

    class CodeMMLU:
        NAME = "CodeMMLU"
        class Tasks:
            CODE_COMPLETION = "code_completion"

    class BigCodeBench:
        NAME = "BigCodeBench"

    class CruxEval:
        NAME = "CruxEval"

CodeMMLU = Benchmarks.CodeMMLU
HumanEval = Benchmarks.HumanEval
BigCodeBench = Benchmarks.BigCodeBench
CruxEval = Benchmarks.CruxEval

class Tasks:
    class CodeGeneration:
        NAME = "code_generation"
        BENCHMARKS = (Benchmarks.HumanEval.NAME, Benchmarks.BigCodeBench.NAME)
        MUTATIONS = [getattr(LexicalMutations, m) for m in dir(LexicalMutations) if not m.startswith("__")]
            

    class MCQInconsistency:
        NAME = "mcq_inconsistency"
        BENCHMARKS = (Benchmarks.CodeMMLU.NAME,)
        MUTATIONS = [
            getattr(SyntacticMutations, m) for m in dir(SyntacticMutations) if not m.startswith("__")] + [
            getattr(LogicalMutations, m) for m in dir(LogicalMutations) if not m.startswith("__")] + [
            getattr(LexicalMutations, m) for m in dir(LexicalMutations) if not m.startswith("__")
            ]
        
    class OutputPrediction:
        NAME = "output_prediction"
        BENCHMARKS = (Benchmarks.HumanEval.NAME, Benchmarks.CruxEval.NAME)
        MUTATIONS = [
            getattr(SyntacticMutations, m) for m in dir(SyntacticMutations) if not m.startswith("__")] + [
            getattr(LogicalMutations, m) for m in dir(LogicalMutations) if not m.startswith("__")] + [
            getattr(LexicalMutations, m) for m in dir(LexicalMutations) if not m.startswith("__")
            ]

    class InputPrediction(OutputPrediction):
        NAME = "input_prediction"

CodeGeneration = Tasks.CodeGeneration
MCQInconsistency = Tasks.MCQInconsistency
OutputPrediction = Tasks.OutputPrediction
InputPrediction = Tasks.InputPrediction

class PromptConfig:
    def __init__(self, prompt_helper: Callable, example_helper: Callable | None):
        self.prompt_helper = prompt_helper
        self.example_helper = example_helper

CODE_INCONSISTENCY_PROMPT_CONFIG ={
    "general" : {
        "zero_shot" : PromptConfig(
            prompt_helper = PredictionInconsistencyPromptTemplate.OutputPrediction.zero_shot_prompt, 
            example_helper = None
            ),
        "one_shot": PromptConfig(
            prompt_helper = PredictionInconsistencyPromptTemplate.OutputPrediction.one_shot_prompt, 
            example_helper = PredictionInconsistencyPromptTemplate.structure_one_shot_example
            ),
        "few_shot" : PromptConfig(
            prompt_helper = PredictionInconsistencyPromptTemplate.OutputPrediction.few_shot_prompt,
            example_helper = PredictionInconsistencyPromptTemplate.structure_few_shot_examples
        ),
    },
    "llama": {
        "zero_shot" : {},
        "one_shot": {},
        "few_shot" : {},
        
    }
}