from mcq_inconsistency.utility.codemmlu_helper import CodeGenerationCodeMMLUHelper
from code_generation.code_generation_tester import CodeGenerationTester
from code_mutation.mutation_functions import CodeMutator, InvalidIteratorError, NoForLoopError
from utility.constants import PromptTypes, Tasks, MCQInconsistency, CodeMMLU, NonReasoningModels, ReasoningModels
from typing import Callable, Dict, Any, List
from tqdm import tqdm
import ast
from code_mutation.mutation_relations import check_for_mutation_conflicts
from llm_models.gpu_code_llms import TransformersCodeLLM
from mcq_inconsistency.prompt_templates.prompt_template import MCQInconsistencyPromptTemplate
import torch

ANS_DICT = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
}

class LLMMCQInconsistencyTester(CodeGenerationTester):
    def __init__(self, qn_database: str = "HumanEval_Input_Output"):
        super().__init__(qn_database=qn_database)

    def process_llm_ans(prog: str) -> Any:
        try:
            result = ast.literal_eval(prog)
            # If the result is a tuple, extract the first element (the actual answer)
            if isinstance(result, tuple) and len(result) >= 1:
                return result[0]
            return result
        except Exception:
            return prog.strip('"').strip("'") if isinstance(prog, str) else prog        

    def run_mcq_inconsistency_test(
            self,
            prompt_helper: Callable[[], str], 
            output_file_path: str,
            prompt_type: str,
            task_set: str,
            num_tests: int,
            continue_from_task: str = None,
            mutations: List[str] = None,
            example_helper: Callable[[Dict[str, str]], str] = None,
            task_type: str = Tasks.OutputPrediction,
            model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    ) -> int:
        # integer storing the number of seconds that the llm should return its answer by
        llm_timeout = 20
                
        if prompt_type == PromptTypes.ONE_SHOT:
            example_helper= MCQInconsistencyPromptTemplate.structure_one_shot_example
        elif prompt_type == PromptTypes.FEW_SHOT:
            example_helper = MCQInconsistencyPromptTemplate.structure_few_shot_examples
        else:
            example_helper = None

        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('MCQ')[-1])
        else:
            continue_from = 0

        num_tests = min(self.question_database.count_documents({}) - continue_from, num_tests)         # ensuring that the number of iterations is lower than max number of documents in the db
        
        if not check_for_mutation_conflicts(mutations=mutations):
            raise ValueError("An invalid combination of mutations were used.")

        for mutation in mutations:
            if mutation not in MCQInconsistency.MUTATIONS:
                raise ValueError(f"{mutation} mutation is an invalid mutation for mcq inconsistency.")
            
        if task_set not in MCQInconsistency.BENCHMARKS:
            raise ValueError(f"{task_set} is an invalid benchmark dataset for mcq inconsistency. Only {MCQInconsistency.BENCHMARKS} datasets are valid.")

        valid_task_types = [getattr(CodeMMLU.Tasks, t) for t in dir(CodeMMLU.Tasks) if not t.startswith("__")]
        if task_type not in valid_task_types:
            raise ValueError(f"{task_type} is an invalid benchmark dataset for mcq inconsistency. Only {valid_task_types} task types are valid.")

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions

        using_GPU = True if (torch.cuda.is_available() or LLMMCQInconsistencyTester.is_colab()) else False
        if using_GPU:
            answers = self.question_database.find({}, { "_id": 0, "answer": 1 })
            filtered_ans = [ans['answer'] for ans in answers]
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
                task_id = f"{task_set}MCQ{idx}"

                qn_sample = self.question_database.find_one({"_id": task_id})
                if qn_sample is None:                               # next task if unable to extract the specific qn id from MongoDB
                    print(f"Document {task_id} not found in database")
                    continue

                prompt_template = prompt_helper()
            
                question = qn_sample['question']                    # task question, which does not inclue the doc string descriptions
                qn_desc = qn_sample['qn_desc']                      # task description. This should be the extracted doc string from the original task
                examples = qn_sample['examples']                    # examples for other prompt techniques like one shot, few shot

                check_function = qn_sample['check']                 # test suite for testing the full solution

                func_name = qn_sample['func_name']                  # function name for the task
                choices = qn_sample['choices']                      # MCQ options for the task
                answer = qn_sample['answer']                        # MCQ answer for the task
                
                ## Dicionary containing the log entry
                log_entry = {
                    "task_id": task_id,
                    "prompt": None,
                    "model_output": None,
                    "correct_answer": answer,
                    "failure_type": None
                }

                ## Sanity check ensuring that the tasks fulfill the minimum requirements for each prompt type.
                if prompt_type == PromptTypes.ONE_SHOT and len(examples.keys()) < 1:
                    log_entry['failure_type'] = 'InsufficientExamplesError > Less than 1 example provided, invalid task for one shot prompting'
                    LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue
                elif prompt_type == PromptTypes.FEW_SHOT and len(examples.keys()) <= 1:
                    log_entry['failure_type'] = 'InsufficientExamplesError > Less than 2 example provided, invalid task for few shot prompting'
                    LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue
                
                # obtaining the correct choice
                correct_choice = choices[answer]

                # formulating the full solution with the correct choice
                full_sol = question + "\n" + correct_choice
                                        
                ## Sanity Check to ensure that the complete solution passes the check functions
                check_soln_validity = CodeGenerationCodeMMLUHelper.check_test_case(
                    test_case = check_function,
                    code_snippet= full_sol,
                    func_name=func_name,
                )
                
                if check_soln_validity is not True:
                    log_entry['failure_type'] = "invalid_full_solution"
                    LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    failed_validity.append(task_id)
                    print(f"Skipping {task_id} as the complete solution did not pass the check function.")
                    continue

                ## Instantiating a codemutator object
                codemutator = CodeMutator(
                    func_name=func_name, 
                    mutated_dict= {
                        'question': question,
                        'full_sol' : full_sol,
                        'choices': choices,
                        'qn_desc': qn_desc,
                        'examples': examples,
                        'check_function': check_function
                    },
                    benchmark_set = task_set
                )

                codemutator.correct_ans_idx = answer
                
                ## Handling Task Mutation (If any)
                try: 
                    for mutation in mutations:
                        codemutator.mutate_for_mcq_inconsistency(
                            mutation_type=mutation,
                            task_set="CodeMMLU",
                            answer=answer,
                            task_type = task_type,
                        )

                except Exception as e:
                    if isinstance(e, (InvalidIteratorError, NoForLoopError)):
                        log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                    else:
                        log_entry['failure_type'] = MutationFailedError(e)
                    LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue

                structured_choices = CodeGenerationCodeMMLUHelper.structure_mcq_choices(choices=codemutator.mutated_dict['choices'])

                ## Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(codemutator.mutated_dict['examples'])
                
                ## Dictionary containing input variables to format the prompt with
                input_variables = {
                    'qn_desc': codemutator.mutated_dict['qn_desc'],
                    'task': codemutator.mutated_dict['question'],
                    'choices': structured_choices,
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
                    ans = LLMMCQInconsistencyTester.process_llm_ans(ans)

                    prob = ans_dict['geom_mean_prob']

                    log_entry['model_output'] = (ans, type(ans))                                            # storing model answer into the database entry
                    log_entry['geometric'] = prob
                else: 
                    try:
                        ans = self.execute_llm(
                            input_variables=input_variables,
                            prompt_template=prompt_template,
                            llm_model=llm,
                            model_name = model_name
                        )
                    except Exception as e:
                        log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                        LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                        continue

                    ans = LLMMCQInconsistencyTester.process_llm_ans(ans)
                    log_entry['model_output'] = (ans, type(ans))                                            # storing model answer into the database entry

                ## Running the formatted prompt into the LLM
                try:
                    assert ans == answer
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
                LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)

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
    llm_tester = LLMMCQInconsistencyTester("HumanEval_Open_Ended")