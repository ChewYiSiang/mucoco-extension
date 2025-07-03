from database import MongoDBHelper
from llm_models.code_llms import CodeLLM
from code_generation.utility.humaneval_functions import CodeGenerationHumanEvalHelper
from utility.mutation_functions import CodeMutator
from typing import Callable, Dict
from tqdm import tqdm
import os
import ast
import random
import time
import pandas as pd
import builtins

class CodeGenerationTester:
    def __init__(self, qn_database: str = "HumanEval_Open_Ended", tf_database: str = "HumanEval_Input_Output"):
        db = MongoDBHelper()
        if db.check_database_connectivity():
            print("MongoDB connected")
        base_qns_db = db.client["Base_Questions_DB"]
        self.question_database = base_qns_db[qn_database]
        self.tf_question_database = base_qns_db[tf_database]
            
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
        
        try:                            # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for idx in tqdm(range(continue_from, num_tests)):
                task_id = f"HumanEvalo{idx}"
                failure_type = None

                qn_sample = self.question_database.find_one({"_id": task_id})

                if qn_sample is None:                       # next task if unable to extract the specific qn id from MongoDB
                    continue

                prompt_template = prompt_helper()
            
                qn = qn_sample['qn']                        # contains the question without the docstring description
                qn_desc = qn_sample['qn_desc']              # doc string description
                examples = qn_sample['examples']            # dict object containing all examples pertaining the question for few shot/one shot prompting
                test_function = qn_sample['check']          # check function for testing validity of a solution
                canon_soln = qn_sample['canon_solution']    # canonical solution to the question
                complete_soln = qn + '\n' + canon_soln      # complete working solution combined from the qn and canon solution

                # Obtaining the function name of the task function
                random_test_case = random.choice(list(examples.keys()))
                func_name = CodeGenerationHumanEvalHelper.extract_func_name_from_example(random_test_case)

                # Sanity Check to ensure that the complete solution passes the check functions
                check_soln_validity = CodeGenerationHumanEvalHelper.check_test_case(test_case = test_function, code_snippet = complete_soln, func_name = func_name)
                
                if check_soln_validity is not True:
                    failed_validity.append(task_id)
                    print(f"Skipping {task_id} as the complete solution did not pass the check function.")
                    continue
                
                # Handling Task Mutation (If any)
                if mutation_type is not None:
                    func_names, var_names = CodeMutator.obtain_key_info_from_code(qn)
                    qn, examples, qn_desc = CodeMutator.mutate_variable_names(
                        source=qn, 
                        qn_desc= qn_desc,
                        examples=examples, 
                        func_names=func_names, 
                        mutation_type=mutation_type,
                        var_names=var_names,
                    )
                
                # Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(examples)
                
                
                input_variables = {
                    'code': qn,
                    'task': qn_desc,
                    'example': prompt_examples if example_helper is not None else None,
                }

                # Running the llm on the input variables and the prompt template
                ans =  llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
                
                try: 
                    # Processing of the llm answer. Some llm answers are in Python code blocks, which needs to be processed as it will fail exec()
                    processed_output = llm.process_ans(ans)
                except ValueError:                          # Raised when the llm answer did not have a python code block
                    try: 
                        exec(ans)                           # Attempting to run the llm answer directly. In some cases, the returned answer can be directly run
                        processed_output = ans              
                    except Exception as e:                  # Else, if the answer is not in a valid code block and cannot be run directly, it is a faulty answer and is stored accordingly.
                        print(f"Could not process LLM answer: {e}")
                        input_data = {
                            "task_id": task_id,
                            "prompt" : prompt_template.format(**input_variables),
                            "model_output": ans,
                            "check_function": None,
                            "canonical_solution": None,
                            "failure_type": type(e)
                        }
                        file_exists = os.path.isfile(output_file_path)

                        # Append the row with or without headers
                        with open(output_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                            df = pd.DataFrame([input_data])
                            df.to_csv(csvfile, header=not file_exists, index=False)
                        continue


                # LLM Answer Test Execution
                try:
                    namespace = {}
                    exec(processed_output, namespace)
                    exec(test_function, namespace)
                    namespace['check'](namespace[func_name])
                    task_pass_count += 1
                    # print(f"{task_id}: {task_pass_count}")

                except Exception as e:
                    failure_type = type(e)
                    if isinstance(e, AssertionError):
                        print("{task_id}: Function failed to run due to following error -> {e}".format(e = e, task_id = task_id))
                    else:
                        print("{task_id}: Could not run the LLM answer due to the following error {e}".format(e = e, task_id = task_id))
                    
                input_data = {
                    "task_id": task_id,
                    "prompt": prompt_template.format(**input_variables),
                    "model_output": processed_output,
                    "check_function": test_function,
                    "canonical_solution": complete_soln,
                    "failure_type": failure_type
                    
                }
                file_exists = os.path.isfile(output_file_path)

                # Append the row with or without headers
                with open(output_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                    df = pd.DataFrame([input_data])
                    df.to_csv(csvfile, header=not file_exists, index=False)

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
