from prediction_inconsistency.utility.humaneval_helper import PredictionInconsistencyHumanEvalHelper
from code_generation.code_generation_tester import CodeGenerationTester
from code_mutation.mutation_functions import CodeMutator
from utility.constants import PromptTypes, Tasks, InputPrediction, MCQInconsistency, CruxEval, HumanEval, ReasoningModels, NonReasoningModels
from typing import Callable, Dict, Any, List
from tqdm import tqdm
import ast
import copy
import torch
import multiprocessing
from llm_models.gpu_code_llms import TransformersCodeLLM
from code_mutation.mutation_relations import check_for_mutation_conflicts
from prediction_inconsistency.prompt_templates.prompt_template import PredictionInconsistencyPromptTemplate


def invoke_llm(input_variables: Dict[str, str], prompt_template: str, queue: multiprocessing.Queue, llm_model : Callable):
    llm = llm_model()
    ans = llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
    queue.put(ans)

class LLMConsistencyTester(CodeGenerationTester):
    def __init__(self, qn_database: str = "HumanEval_Input_Output", base_db : str = "Base_Questions_DB", n: int = 2):
        super().__init__(qn_database=qn_database, base_db= base_db, n=n)
    
    def process_llm_ans(prog: str) -> Any:
        try:
            result = ast.literal_eval(prog)
            # If the result is a tuple, extract the first element (the actual answer)
            if isinstance(result, tuple) and len(result) >= 1:
                return result[0]
            return result
        except Exception:
            return prog.strip('"').strip("'") if isinstance(prog, str) else prog

    def run_code_consistency_test(
            self,
            prompt_helper: Callable[[], str], 
            output_file_path: str,
            prompt_type: str,
            task_set: str,
            num_tests: int = None,
            continue_from_task: str = None,
            mutations: List[str] = [],
            example_helper: Callable[[Dict[str, str]], str] = None,
            task_type: str = Tasks.OutputPrediction.NAME,
            model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ) -> int:
        # integer storing the number of seconds that the llm should return its answer by
        llm_timeout = 20

        if prompt_type == PromptTypes.ONE_SHOT:
            example_helper= PredictionInconsistencyPromptTemplate.InputPrediction.structure_one_shot_example
        elif prompt_type == PromptTypes.FEW_SHOT:
            example_helper = PredictionInconsistencyPromptTemplate.InputPrediction.structure_few_shot_examples
        else:
            example_helper = None

        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('TF')[-1])
        else:
            continue_from = 0

        num_tests = min(self.question_database.count_documents({}) - continue_from, num_tests)         # ensuring that the number of iterations is lower than max number of documents in the db
        
        if not check_for_mutation_conflicts(mutations=mutations):
            raise ValueError("An invalid combination of mutations were used.")

        for mutation in mutations:
            if mutation not in InputPrediction.MUTATIONS:
                raise ValueError(f"{mutation} mutation is an invalid mutation for prediction inconsistency.")
            
        if task_set not in InputPrediction.BENCHMARKS:
            raise ValueError(f"{task_set} is an invalid benchmark dataset for prediction inconsistency. Only {MCQInconsistency.BENCHMARKS} datasets are valid.")

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions
        using_GPU = True if (torch.cuda.is_available() or LLMConsistencyTester.is_colab()) else False

        if using_GPU:
            if task_type == Tasks.OutputPrediction.NAME:
                answers = self.question_database.find({}, { "_id": 0, "output": 1 })
                filtered_ans = [ans['output']['args'] for ans in answers]
            else:
                filtered_ans = ["True"]

            llm = TransformersCodeLLM(model_name=model_name, answers= filtered_ans)
        else:
            reasoning_models = [getattr(ReasoningModels, model) for model in dir(ReasoningModels) if not model.startswith("_")]
            non_reasoning_models = [getattr(NonReasoningModels, model) for model in dir(NonReasoningModels) if not model.startswith("_")]
            all_local_models = reasoning_models + non_reasoning_models
            for model in all_local_models:
                if model['name'] == model_name:
                    llm = model['model_class']
                    break
            else:
                valid_model_names = [model['name'] for model in all_local_models]
                raise ValueError(f"{model_name} is not a valid local model. The models supported by this framework are {', '.join(valid_model_names)}")
        try:                            # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for idx in tqdm(range(continue_from, continue_from + num_tests)):
                task_id = f"{task_set}TF{idx}"

                qn_sample = self.question_database.find_one({"_id": task_id})
                if qn_sample is None:                               # next task if unable to extract the specific qn id from MongoDB
                    print(f"Document {task_id} not found in database")
                    continue

                prompt_template = prompt_helper()
                full_sol = qn_sample['full_sol']                    # full canonical solution for the task
                qn_desc = qn_sample.get('qn_desc', "")              # task description. This should be the extracted doc string from the original task
                examples = qn_sample.get('examples', {})            # examples for other prompt techniques like one shot, few shot

                test_inputs = qn_sample['input']                    # unpacking input args and metadata from qn
                input_args = test_inputs['args']                    # test input args
                input_metadata = test_inputs['metadata']            # test input metadata

                test_outputs = qn_sample['output']                  # unpacking outputs args and metadata from qn
                output_args = test_outputs['args']                  # test output args
                output_metadata = test_outputs['metadata']          # test output metadata

                if task_set == HumanEval.NAME:
                    func_name = qn_sample.get("func_name", None)
                elif task_set == CruxEval.NAME:
                    func_name = "f"
                else:
                    ValueError("This task_set has not been implemented yet")
                
                if output_metadata == type(None).__name__:
                    output_metadata = "type(None)"
                
                if not isinstance(output_args, str) and not isinstance(eval(str(output_args)), eval(output_metadata)):
                    if eval(output_metadata) == tuple:
                        output_args = tuple(output_args)

                input_args = eval(input_args) if isinstance(input_args, str) and input_metadata != str.__name__  else input_args

                ## Dicionary containing the log entry
                log_entry = {
                    "task_id": task_id,
                    "prompt": None,
                    "model_output": None,
                    "expected_output": test_outputs if task_type == Tasks.OutputPrediction.NAME else "True",
                    "failure_type": None
                }

                ## Sanity check ensuring that the tasks fulfill the minimum requirements for each prompt type.
                if prompt_type == PromptTypes.ONE_SHOT and len(examples.keys()) < 1:
                    log_entry['failure_type'] = 'InsufficientExamplesError > Less than 1 example provided, invalid task for one shot prompting'
                    LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue

                elif prompt_type == PromptTypes.FEW_SHOT and len(examples.keys()) <= 1:
                    log_entry['failure_type'] = 'InsufficientExamplesError > Less than 2 example provided, invalid task for few shot prompting'
                    LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue
                
                ## Processing of output args and metadata
                output_args = ast.literal_eval(output_args) if output_metadata != str.__name__ else output_args
                                        
                ## Sanity Check to ensure that the complete solution passes the check functions
                check_soln_validity = PredictionInconsistencyHumanEvalHelper.check_input_output(
                    full_sol= full_sol,
                    test_input= copy.deepcopy(input_args),
                    expected_output= output_args,
                    func_name=func_name,
                    input_metadata = input_metadata
                )
                
                if check_soln_validity is not True:
                    log_entry['failure_type'] = "invalid_full_solution"
                    LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    failed_validity.append(task_id)
                    print(f"Skipping {task_id} as the complete solution did not pass the check function.")
                    continue

                ## Instantiating a codemutator object
                codemutator = CodeMutator(
                    func_name=func_name,
                    mutated_dict={
                        "question": full_sol,
                        "full_sol": full_sol,
                        "qn_desc": qn_desc,
                        "examples": examples,
                    },
                    benchmark_set=task_set
                )
                
                ## Handling Task Mutation (If any)
                try: 
                    for mutation_type in mutations:
                        codemutator.mutate_for_prediction_inconsistency_test(
                            mutation_type = mutation_type,
                            input_args= copy.deepcopy(input_args),
                            output_args= output_args,
                            input_metadata= input_metadata,
                            task_set = task_set
                        )
                        
                except Exception as e:
                    # If no mutation was requested, treat as unexpected error and continue
                    log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                    LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue

                ## Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(codemutator.mutated_dict.get('examples', None))
                
                ## Dictionary containing input variables to format the prompt with
                input_variables = {
                    'qn_desc': codemutator.mutated_dict.get('qn_desc', None),
                    'full_sol': codemutator.mutated_dict.get('full_sol', None),
                    'test_input': f'"{input_args}"' if isinstance(input_args, str) else input_args,
                    'test_output': f'"{output_args}"' if isinstance(output_args, str) else output_args,
                    'example': prompt_examples if example_helper is not None else None,
                }
                log_entry["prompt"] = prompt_template.format(**input_variables)            # storing formatted prompt into database entry
                
                ## Running the llm on the input variables and the prompt template
                if using_GPU:
                    ans_dict = llm.invoke(
                        input_variables=input_variables,
                        prompt_template=prompt_template
                    )

                    ans = ans_dict['ans']
                    ans = LLMConsistencyTester.process_llm_ans(ans)
                    log_entry['model_output'] = (ans, type(ans))                                            # storing model answer into the database entry
                    
                    if task_type == InputPrediction.NAME:
                        log_entry['geometric'] = ans_dict["geom_mean_prob"]
                else: 
                    try:
                        ans = self.execute_llm(
                            input_variables=input_variables,
                            prompt_template=prompt_template,
                            llm_model=llm,
                            model_name=model_name
                        )
                    except Exception as e:
                        log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                        LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                        continue

                    ans = LLMConsistencyTester.process_llm_ans(ans)
                
                    log_entry['model_output'] = (ans, type(ans))                                            # storing model answer into the database entry
                    
                ## Running the formatted prompt into the LLM
                try:
                    if task_type == Tasks.OutputPrediction.NAME:
                        assert ans == output_args
                    elif task_type == Tasks.InputPrediction.NAME:
                        assert ans == True
                except Exception as e:
                    if isinstance(e, AssertionError):
                        pass
                        # print(f"{task_id}: Function failed to run due to following error: {type(e)} > {e}")
                    elif isinstance(e, RuntimeError):
                        e = RuntimeError(f"LLM did not complete answering the question within the given timeout of {llm_timeout} seconds")
                    else:
                        print(f"{task_id}: Could not run the LLM answer due to the following error: {type(e)} > {e}")
                    log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                
                ## Logging data into the csv file
                LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)

            return task_pass_count
        
        except Exception as e:
            print(type(e))
            print(e)
            print(task_id)
            return task_pass_count
        
        except KeyboardInterrupt:
            print(task_id)
            return task_pass_count

class MutationFailedError(Exception):
    def __init__(self, error):
        super().__init__(f"Mutation failed due to the following error: {type(error).__name__} > {error}")

if __name__ == "__main__":
    llm_tester = LLMConsistencyTester("HumanEval_Open_Ended")
