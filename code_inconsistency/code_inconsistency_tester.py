from code_inconsistency.utility.humaneval_functions import CodeInconsistencyHumanEvalHelper
from code_generation.code_generation_tester import CodeGenerationTester
from code_mutation.mutation_functions import CodeMutator
from llm_models.code_llms import CodeLLM
from typing import Callable, Dict, Any
from tqdm import tqdm
import time
import ast



class LLMConsistencyTester(CodeGenerationTester):
    def __init__(self, qn_database: str = "HumanEval_Input_Output"):
        super().__init__(qn_database=qn_database)

    def process_llm_ans(prog: str) -> Any:
        try:
            return ast.literal_eval(prog)
        except Exception:
            return prog.strip('"').strip("'") if isinstance(prog, str) else prog

            
    def run_code_consistency_test(
            self,
            llm: CodeLLM,
            prompt_helper: Callable[[], str], 
            num_tests: int, 
            output_file_path: str,
            prompt_type: str,
            continue_from_task: str = None,
            lexical_mutation: str = None,
            syntactic_mutation: str = None,
            example_helper: Callable[[Dict[str, str]], str] = None, 
    ) -> int:
        
        if prompt_type != 'zero_shot' and example_helper is None:
            raise ValueError("A non zero-shot prompt is used, yet no example helper function was given. Add the approrpriate example_helper for this prompt template.")
        
        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('TF')[-1])
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

        num_tests = min(num_tests, self.question_database.count_documents({}) - continue_from)         # ensuring that the number of iterations is lower than max number of documents in the db

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions
        try:                            # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for idx in tqdm(range(continue_from, continue_from + num_tests)):
                task_id = f"HumanEvalTF{idx}"

                qn_sample = self.question_database.find_one({"_id": task_id})
                if qn_sample is None:                               # next task if unable to extract the specific qn id from MongoDB
                    continue

                prompt_template = prompt_helper()
            
                full_sol = qn_sample['full_sol']                    # full canonical solution for the task
                qn_desc = qn_sample['qn_desc']                      # task description. This should be the extracted doc string from the original task
                examples = qn_sample['examples']                    # examples for other prompt techniques like one shot, few shot

                test_inputs = qn_sample['input']                    # unpacking input args and metadata from qn
                input_args = test_inputs['args']                    # test input args
                input_metadata = test_inputs['metadata']            # test input metadata

                test_outputs = qn_sample['output']                  # unpacking outputs args and metadata from qn
                output_args = test_outputs['args']                  # test output args
                output_metadata = test_outputs['metadata']          # test output metadata
                

                if output_metadata == type(None).__name__:
                    output_metadata = "type(None)"
                if not isinstance(output_args, str) and not isinstance(eval(str(output_args)), eval(output_metadata)):
                    if eval(output_metadata) == tuple:
                        output_args = tuple(output_args)

                ## Dicionary containing the log entry
                log_entry = {
                    "task_id": task_id,
                    "prompt": None,
                    "model_output": None,
                    "expected_output": test_outputs,
                    "failure_type": None
                }
                                        
                ## Sanity Check to ensure that the complete solution passes the check functions
                check_soln_validity = CodeInconsistencyHumanEvalHelper.check_database_answer(
                    full_sol= full_sol,
                    input_args=input_args,
                    input_metadata=input_metadata,
                    output_args= output_args,
                    output_metadata= output_metadata,
                    examples = examples
                    )
                ## Processing of output args and metadata
                output_args = ast.literal_eval(output_args) if output_metadata != str.__name__ else output_args
                
                if check_soln_validity is not True:
                    log_entry['failure_type'] = "invalid_full_solution"
                    LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    failed_validity.append(task_id)
                    print(f"Skipping {task_id} as the complete solution did not pass the check function.")
                    continue
                
                ## Handling Task Mutation (If any)
                try: 
                    for mutation_type in mutation_dict.values():
                        mutated_dict = CodeMutator.mutate_for_code_inconsistency_test(
                            mutation_type = mutation_type,
                            full_sol = full_sol,
                            examples= examples,
                            qn_desc= qn_desc,
                            input_args= input_args,
                            output_args= output_args
                        )

                        full_sol = mutated_dict['full_sol']
                        qn_desc = mutated_dict['qn_desc']
                        examples = mutated_dict['examples']

                except Exception as e:
                    log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                    LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)
                    continue

                ## Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(examples)
                
                ## Dictionary containing input variables to format the prompt with
                input_variables = {
                    'qn_desc': qn_desc,
                    'full_sol': full_sol,
                    'test_input': f'"{input_args}"' if isinstance(input_args, str) else input_args,
                    'example': prompt_examples if example_helper is not None else None,
                }
                log_entry["prompt"] = prompt_template.format(**input_variables)            # storing formatted prompt into database entry

                ## Running the llm on the input variables and the prompt template
                ans =  llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
                ans = LLMConsistencyTester.process_llm_ans(ans)
                log_entry['model_output'] = (ans, type(ans))                                            # storing model answer into the database entry
                ## Running the formatted prompt into the LLM
                try:
                    assert ans == output_args
                except Exception as e:
                    if isinstance(e, AssertionError):
                        pass
                        # print(f"{task_id}: Function failed to run due to following error: {type(e)} > {e}")
                    else:
                        print(f"{task_id}: Could not run the LLM answer due to the following error {type(e)} > {e}")
                    log_entry['failure_type'] = f"{type(e).__name__} > {e}"
                
                ## Logging data into the csv file
                LLMConsistencyTester.log_into_csv(output_file_path = output_file_path, input_data = log_entry)

                time.sleep(2)

            return task_pass_count
        
        # except Exception as e:
        #     print(type(e))
        #     print(e)
        #     print(task_id)
        #     return task_pass_count
        
        except KeyboardInterrupt:
            print(task_id)
            return task_pass_count

if __name__ == "__main__":
    llm_tester = LLMConsistencyTester("HumanEval_Open_Ended")
