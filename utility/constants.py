from llm_models.code_llms import OpenAILLM, Codestral, DeepSeekLLM
from llm_models.code_reasoning_llms import OpenAIReasoningLLM
from llm_models.gpu_code_llms import TransformersCodeLLM

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

    class Turbulence:
        NAME = "Turbulence"

CodeMMLU = Benchmarks.CodeMMLU
HumanEval = Benchmarks.HumanEval
BigCodeBench = Benchmarks.BigCodeBench
CruxEval = Benchmarks.CruxEval
Turbulence = Benchmarks.Turbulence

class Tasks:
    class CodeGeneration:
        NAME = "code_generation"
        BENCHMARKS = (HumanEval.NAME, BigCodeBench.NAME, Turbulence.NAME)
        MUTATIONS = [getattr(LexicalMutations, m) for m in dir(LexicalMutations) if not m.startswith("__")]
            

    class MCQInconsistency:
        NAME = "mcq_inconsistency"
        BENCHMARKS = (CodeMMLU.NAME,)
        MUTATIONS = [
            getattr(SyntacticMutations, m) for m in dir(SyntacticMutations) if not m.startswith("__")] + [
            getattr(LogicalMutations, m) for m in dir(LogicalMutations) if not m.startswith("__")] + [
            getattr(LexicalMutations, m) for m in dir(LexicalMutations) if not m.startswith("__")
            ]
        
    class OutputPrediction:
        NAME = "output_prediction"
        BENCHMARKS = (HumanEval.NAME, CruxEval.NAME, Turbulence.NAME)
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

class LLMModels:
    class ReasoningModels:
        GPT5 = {"name": "gpt-5", "model_class": OpenAIReasoningLLM}
    
    class NonReasoningModels:
        CODESTRAL = {"name": "codestral-latest", "model_class": Codestral}
        GPT4O = {"name": "gpt-4o", "model_class": OpenAILLM}
        DEEPSEEK_CHAT = {"name": "deepseek-chat", "model_class": DeepSeekLLM}
        LLAMA_3_1_8B = {"name": "meta-llama/Llama-3.1-8B-Instruct", "model_class": TransformersCodeLLM}


ReasoningModels = LLMModels.ReasoningModels
NonReasoningModels = LLMModels.NonReasoningModels

class SamplingMethods:
    SYSTEMATIC = "systematic"
    RANDOM = 'random'

class Seed:
    value = 1234