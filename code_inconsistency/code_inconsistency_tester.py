from database import MongoDBHelper
from llm_models.code_llms import CodeLLM
from code_inconsistency.utility.humaneval_functions import CodeInconsistencyHumanEvalHelper
from utility.mutation_functions import CodeMutator
from typing import Callable, Dict, Tuple, List
from code_generation.code_generation_tester import CodeGenerationTester
from tqdm import tqdm
import os
import random
import time
import pandas as pd

class LLMConsistencyTester(CodeGenerationTester):
    def __init__(self, qn_database: str = "HumanEval_Input_Output"):
        super().__init__(qn_database=qn_database)
            
    def run_code_consistency_test(
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
            continue_from = int(continue_from_task.split('TF')[-1])
        else:
            continue_from = 0

        num_tests = min(num_tests, self.question_database.count_documents({}))         # ensuring that the number of iterations is lower than max number of documents in the db

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions
        try:                            # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for idx in tqdm(range(continue_from, num_tests)):
                task_id = f"HumanEvalTF{idx}"
                failure_type = None

                qn_sample = self.question_database.find_one({"_id": task_id})

                if qn_sample is None:                       # next task if unable to extract the specific qn id from MongoDB
                    continue

                prompt_template = prompt_helper()
            
                full_sol = qn_sample['full_sol']  
                qn_desc = qn_sample['qn_desc']                      
                inputs = qn_sample['input']
                test_input = inputs['test_input']
                input_metadata = inputs['input_metadata']
                examples = qn_sample['examples']              
                expected_output = qn_sample['expected_output']      
                
                # Obtaining the function name of the task function
                random_test_case = random.choice(list(examples.keys()))
                func_name = CodeInconsistencyHumanEvalHelper.extract_func_name_from_example(random_test_case)      
                
                # Sanity Check to ensure that the complete solution passes the check functions
                check_soln_validity = CodeInconsistencyHumanEvalHelper.check_input_output(
                    full_sol= full_sol,
                    test_input=test_input,
                    expected_output=expected_output,
                    func_name=func_name,
                    input_metadata = input_metadata
                )
                
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
                    'qn_desc': qn_desc,
                    'full_sol': full_sol,
                    'test_input': test_input,
                    'example': prompt_examples if example_helper is not None else None,
                }

                # Running the llm on the input variables and the prompt template
                ans =  llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
                try: 
                    assert eval(ans) == eval(str(expected_output))
                except Exception as e:
                    failure_type = type(e)
                    if isinstance(e, AssertionError):
                        print("{task_id}: Function failed to run due to following error -> {e}".format(e = type(e), task_id = task_id))
                    else:
                        print("{task_id}: Could not run the LLM answer due to the following error {e}".format(e = type(e), task_id = task_id))
                    
                input_data = {
                    "task_id": task_id,
                    "prompt": prompt_template.format(**input_variables),
                    "model_output": ans,
                    "expected_output": expected_output,
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
    llm_tester = LLMConsistencyTester("HumanEval_Open_Ended")
