from mcq_inconsistency.utility.codemmlu_helper import CodeGenerationCodeMMLUHelper
from code_generation.code_generation_tester import CodeGenerationTester
from code_mutation.mutation_functions import CodeMutator
from utility.constants import PromptTypes, Tasks
from typing import Callable, Dict, Any, List
from tqdm import tqdm
import time
import ast
import multiprocessing
from llm_models.code_llms import Mistral


ANS_DICT = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
}

def invoke_llm(input_variables: Dict[str, str], prompt_template: str, queue: multiprocessing.Queue):
    llm = Mistral()
    ans = llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
    queue.put(ans)


class LLMMCQInconsistencyTester(CodeGenerationTester):
    def __init__(self, qn_database: str = "HumanEval_Input_Output"):
        super().__init__(qn_database=qn_database)

    def process_llm_ans(prog: str) -> Any:
        try:
            return ast.literal_eval(prog)
        except Exception:
            return prog.strip('"').strip("'") if isinstance(prog, str) else prog        

    def run_mcq_inconsistency_test(
            self,
            prompt_helper: Callable[[], str], 
            output_file_path: str,
            prompt_type: str,
            task_set: str,
            num_tests: int = None,
            continue_from_task: str = None,
            lexical_mutation: str = None,
            syntactic_mutation: str = None,
            example_helper: Callable[[Dict[str, str]], str] = None,
            specific_doc_ids: List[str] = None,
            task_type: str = Tasks.CodeInconsistency.OutputPrediction,
    ) -> int:
        # integer storing the number of seconds that the llm should return its answer by
        llm_timeout = 20
                
        if prompt_type != 'zero_shot' and example_helper is None:
            raise ValueError("A non zero-shot prompt is used, yet no example helper function was given. Add the approrpriate example_helper for this prompt template.")
        
        if num_tests is None and specific_doc_ids is None:
            raise ValueError("Either num_tests or specific_doc_ids must be provided.")
        
        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('MCQ')[-1])
        else:
            continue_from = 0

        ## Dictionary storing the types of lexical mutation and syntactic mutation
        mutation_dict = {
            "lexical_mutation" : lexical_mutation,
            "syntactic_mutation" : syntactic_mutation
        }

        for mutation in mutation_dict.values():
            if mutation is not None and mutation not in CodeMutator.mutation_types:
                raise ValueError(f"An invalid type of mutation is used. Only {CodeMutator.mutation_types} type of mutations are valid.")

        # Determine which documents to test
        if specific_doc_ids is not None:
            # Use specific document IDs
            if num_tests is not None:
                test_docs = specific_doc_ids[:num_tests]  # Limit to num_tests if specified
            else:
                test_docs = specific_doc_ids  # Use all provided documents
            print(f"Testing {len(test_docs)} specific documents")
        else:
            # Use original sequential approach
            if num_tests is None:
                raise ValueError("num_tests must be provided when specific_doc_ids is not used.")
            num_tests = min(num_tests, self.question_database.count_documents({}) - continue_from)
            test_docs = [f"HumanEvalTF{idx}" for idx in range(continue_from, continue_from + num_tests)]
            print(f"Testing documents from HumanEvalTF{continue_from} to HumanEvalTF{continue_from + num_tests - 1}")

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions

        try:                            # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for idx in tqdm(range(continue_from, continue_from + num_tests)):
                task_id = f"{task_set}{idx}"

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
                correct_choice = choices[ANS_DICT[answer]]

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
                codemutator = CodeMutator(func_name=func_name)

                codemutator.mutated_dict = {
                    'question': question,
                    'choices': choices,
                    'qn_desc': qn_desc,
                    'examples': examples,
                    'choices': choices,
                    'check_function': check_function
                }

                codemutator.correct_ans_idx = ANS_DICT[answer]
                
                ## Handling Task Mutation (If any)
                try: 
                    for mutation_type in mutation_dict.values():
                        codemutator.mutate_for_mcq_inconsistency(
                            mutation_type=mutation_type,
                            task_set="CodeMMLU",
                            correct_answer_idx=ANS_DICT[answer],
                            task_type = task_type,
                        )

                except Exception as e:
                    log_entry['failure_type'] = MutationFailedError(e)
                    LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue

                structured_choices = CodeGenerationCodeMMLUHelper.structure_mcq_choices(choices=codemutator.mutated_dict['choices'])

                ## Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(examples)
                
                ## Dictionary containing input variables to format the prompt with
                input_variables = {
                    'qn_desc': codemutator.mutated_dict['qn_desc'],
                    'task': codemutator.mutated_dict['question'],
                    'choices': structured_choices,
                    'example': prompt_examples if example_helper is not None else None,
                }
                log_entry["prompt"] = prompt_template.format(**input_variables)            # storing formatted prompt into database entry

                ## Running the llm on the input variables and the prompt template
                multiprocessing_queue = multiprocessing.Queue()

                verify_answer_process = multiprocessing.Process(
                    target= invoke_llm,
                    kwargs={
                        "input_variables": input_variables,
                        "prompt_template": prompt_template,
                        "queue": multiprocessing_queue
                    }
                )

                verify_answer_process.start()
                verify_answer_process.join(timeout=llm_timeout)

                if verify_answer_process.is_alive():
                    verify_answer_process.kill()
                    verify_answer_process.join()
                    log_entry['failure_type'] = f"{type(RuntimeError()).__name__} > LLM could not answer the task within {llm_timeout} seconds."
                    LLMMCQInconsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue

                if not multiprocessing_queue.empty():
                    ans = multiprocessing_queue.get()

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

                time.sleep(2)


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