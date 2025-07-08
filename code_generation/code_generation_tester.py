from database import MongoDBHelper
from llm_models.code_llms import CodeLLM
from code_generation.utility.humaneval_functions import CodeGenerationHumanEvalHelper
from code_mutation.mutation_functions import CodeMutator
from typing import Callable, Dict, Any
from tqdm import tqdm
import os
import random
import time
import pandas as pd
import multiprocessing

def run_llm_answer(processed_output: str, test_function: str, func_name: str, mp_queue = multiprocessing.Queue):
        namespace = {}
        try:
            exec(processed_output, namespace)
            exec(test_function, namespace)
            namespace["check"](namespace[func_name])
        except Exception as e:
            mp_queue.put(e)

class CodeGenerationTester:
    def __init__(self, qn_database: str = "HumanEval_Open_Ended"):
        db = MongoDBHelper()
        if db.check_database_connectivity():
            print("MongoDB connected")
        base_qns_db = db.client["Base_Questions_DB"]
        self.question_database = base_qns_db[qn_database]
    

    def log_into_csv(output_file_path:str, input_data = Dict[str, Any]) -> None:
        file_exists = os.path.isfile(output_file_path)

        # Append the row with or without headers
        with open(output_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            df = pd.DataFrame([input_data])
            df.to_csv(csvfile, header=not file_exists, index=False)


            
    def run_code_generation_test(
            self, 
            llm: CodeLLM, 
            prompt_helper: Callable[[], str], 
            num_tests: int, 
            output_file_path: str,
            prompt_type: str,
            continue_from_task: str = None,
            mutation_type: str = None,
            example_helper: Callable[[Dict[str, str]], str] = None, 
        ) -> int:
        
        if prompt_type != 'zero_shot' and example_helper is None:
            raise ValueError("A non zero-shot prompt is used, yet no example helper function was given. Add the approrpriate example_helper for this prompt template.")
        
        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('o')[-1])
        else:
            continue_from = 0

        num_tests = min(num_tests, self.question_database.count_documents({}))         # ensuring that the number of iterations is lower than max number of documents in the db

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions
        timeout = 5                     # int variable indicating the number of seconds the LLM generated program should complete running by

        try:                            # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for idx in tqdm(range(continue_from, num_tests)):
                task_id = f"HumanEvalo{idx}"

                qn_sample = self.question_database.find_one({"_id": task_id})

                if qn_sample is None:                                           # skip to the next task if unable to extract the specific qn id from MongoDB
                    continue

                prompt_template = prompt_helper()
            
                qn = qn_sample['qn']                                            # contains the question without the docstring description
                qn_desc = qn_sample['qn_desc']                                  # doc string description
                examples = qn_sample['examples']                                # dict object containing all examples pertaining the question for few shot/one shot prompting
                test_function = qn_sample['check']                              # check function for testing validity of a solution
                canon_soln = qn_sample['canon_solution']                        # canonical solution to the question
                complete_soln = qn + '\n' + canon_soln                          # complete working solution combined from the qn and canon solution

                # Dictionary storing all relevant log data
                input_data = {
                            "task_id": task_id,
                            "prompt" : None,
                            "model_output": None,
                            "check_function": test_function,
                            "canonical_solution": complete_soln,
                            "failure_type": None
                        }

                # Sanity check to filter out test cases where there is only 1 example and hence the task cannot be used for few shot
                if prompt_type == "few_shot" and len(examples.keys()) == 1:
                    print(f"Skipping {task_id} as the complete solution does not have more than 1 example.")
                    input_data['failure_type'] = "insufficient_few_shot_examples"
                    CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data=input_data)
                    continue

                # Obtaining the function name of the task function
                random_test_case = random.choice(list(examples.keys()))
                func_name = CodeGenerationHumanEvalHelper.extract_func_name_from_example(random_test_case)

                # Sanity check to ensure that the complete solution passes the check functions
                check_soln_validity = CodeGenerationHumanEvalHelper.check_test_case(test_case = test_function, code_snippet = complete_soln, func_name = func_name)
                
                if check_soln_validity is not True:
                    failed_validity.append(task_id)
                    print(f"Skipping {task_id} as the complete solution did not pass the check function.")
                    input_data['failure_type'] = "canonical_sol_did_not_pass_check"
                    CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data=input_data)
                    continue
                
                # Handling Task Mutation (If any)
                if mutation_type is not None:
                    func_names, var_names = CodeMutator.obtain_key_info_from_code(qn)
                    qn, examples, qn_desc, mutation_rename_map = CodeMutator.mutate_variable_names(
                        source=qn, 
                        qn_desc= qn_desc,
                        examples=examples, 
                        func_names=func_names, 
                        mutation_type=mutation_type,
                        var_names=var_names,
                    )
                    func_name = mutation_rename_map[func_name]
                
                # Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(examples)
                
                
                input_variables = {
                    'code': qn,
                    'task': qn_desc,
                    'example': prompt_examples if example_helper is not None else None,
                }
                input_data["prompt"] = prompt_template.format(**input_variables)


                # Running the llm on the input variables and the prompt template
                ans =  llm.invoke(input_variables=input_variables, prompt_template=prompt_template)

                ## Processing of LLM Answer
                try: 
                    # Processing of the llm answer. Some llm answers are in Python code blocks, which needs to be processed as it will fail exec()
                    processed_output = llm.process_ans(ans)
                except ValueError:                          # Raised when the llm answer did not have a python code block
                    try: 
                        exec(ans)                           # Attempting to run the llm answer directly. In some cases, the returned answer can be directly run as no code block was returned
                        processed_output = ans              
                    except Exception as e:                  # Else, if the answer is not in a valid code block and cannot be run directly, it is a faulty answer and is stored accordingly.
                        print(f"Could not process LLM answer: {e}")
                        input_data["model_output"] = ans
                        input_data["failure_type"] = ("could_not_parse_LLM_answer", type(e))
                        CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data = input_data)
                        continue
                
                input_data['model_output'] = processed_output       # storing the answer in input_data dict
                ## LLM Answer Test Execution
                try:
                    
                    # multiprocessing library is used here as some LLM answers are wrong and uses a while loop which runs indefinitely.
                    #   This ensures that the LLM answer execution will automatically timeout after timeout seconds
                    multiprocessing_queue = multiprocessing.Queue()
                    verify_answer_process = multiprocessing.Process(target= run_llm_answer, args = (processed_output, test_function, func_name, multiprocessing_queue))
                    verify_answer_process.start()
                    verify_answer_process.join(timeout=timeout)
                    if verify_answer_process.is_alive():
                        verify_answer_process.kill()
                        verify_answer_process.join()
                        raise RuntimeError()
                    if not multiprocessing_queue.empty():
                        error = multiprocessing_queue.get()
                        raise error
                    task_pass_count += 1
                    # print(f"{task_id}: {task_pass_count}")

                except Exception as e:
                    if isinstance(e, AssertionError):
                        print("{task_id}: Function failed to run due to following error -> {e}".format(e = e, task_id = task_id))
                    elif isinstance(e, RuntimeError):
                        print("{task_id}: LLM Answer exceeded runtime of {timeout} seconds -> {e}".format(e = e, task_id = task_id, timeout = timeout))
                    else:
                        print("{task_id}: Could not run the LLM answer due to the following error {e}".format(e = e, task_id = task_id))
                    input_data['failure_type'] = type(e)            # Logging failure type into input_data

                # logging completed run into csv 
                CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data = input_data)

                time.sleep(5)
                
            return task_pass_count
        except Exception as e:
            print(e)
            print(task_id)
            return task_pass_count
        
        except KeyboardInterrupt:
            print(task_id)
            return task_pass_count

    


if __name__ == "__main__":
    llm_tester = CodeGenerationTester("HumanEval_Open_Ended")
