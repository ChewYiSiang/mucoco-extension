from database import MongoDBHelper
from code_generation.utility.humaneval_helper import CodeGenerationHumanEvalHelper
from code_generation.utility.bigcodebench_helper import CodeGenerationBigCodeBenchHelper
from code_mutation.mutation_functions import CodeMutator
from code_generation.prompt_templates.prompt_template import OpenEndedPromptTemplate
from typing import Callable, Dict, Any, List
from tqdm import tqdm
import os
import io
import matplotlib.pyplot as plt
import contextlib
import pandas as pd
import multiprocessing
import shutil
import regex as re
from utility.constants import PromptTypes, CodeGeneration, ReasoningModels, NonReasoningModels, LexicalMutations
from code_mutation.mutation_relations import check_for_mutation_conflicts
from llm_models.code_llms import Mistral
import torch
from llm_models.gpu_code_llms import TransformersCodeLLM


def invoke_llm(input_variables: Dict[str, str], prompt_template: str, queue: multiprocessing.Queue, llm_model : Callable, model_name : str):
    f = io.StringIO()

    llm = llm_model(model_name)

    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        ans = llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
        plt.close('all')
        queue.put(ans)

class Tester:
    def execute_llm(self, model_name: str, input_variables: Dict[str, str], prompt_template: str, llm_model : Callable = Mistral, ):
        llm_timeout = 30
            
        ## Running the llm on the input variables and the prompt template
        multiprocessing_queue = multiprocessing.Queue()
        verify_answer_process = multiprocessing.Process(
            target= invoke_llm,
            kwargs={
                "input_variables": input_variables,
                "prompt_template": prompt_template,
                "queue": multiprocessing_queue,
                "llm_model": llm_model,
                "model_name": model_name,
            }
        )

        verify_answer_process.start()
        verify_answer_process.join(timeout=llm_timeout)

        if verify_answer_process.is_alive():
            verify_answer_process.kill()
            verify_answer_process.join()
            raise LLMExecutionRuntimeError(f"LLM could not answer the task within {llm_timeout} seconds.")

        if not multiprocessing_queue.empty():
            ans = multiprocessing_queue.get()
            return ans
        else:
            raise LLMExecutionError(f"LLM did not return any answer")
        

class CodeGenerationTester(Tester):
    def __init__(self, qn_database: str = "HumanEval_Open_Ended", base_db : str = "Base_Questions_DB", n: int = 2):
        db = MongoDBHelper(n)
        if db.check_database_connectivity():
            print("MongoDB connected")
        base_qns_db = db.client[base_db]
        self.question_database = base_qns_db[qn_database]
    
    @staticmethod
    def is_colab():
        try:
            import google.colab
            return True
        except ImportError:
            return False
        
    @staticmethod
    def log_into_csv(output_file_path:str, input_data = Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        file_exists = os.path.isfile(output_file_path)

        # Append the row with or without headers
        with open(output_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            df = pd.DataFrame([input_data])
            df.to_csv(csvfile, header=not file_exists, index=False)
    

    @staticmethod
    def process_llm_ans(text: str) -> str:
        # Clean chat template tokens
        chat_tokens = ['<|im_start|>', '<|im_end|>', '<|begin_of_text|>', '<|start_header_id|>', 
                      '<|end_header_id|>', '<|eot_id|>', '<|end_of_text|>', 'system', 'user', 'assistant',
                      '<start_of_turn>', '<end_of_turn>', '</end_of_turn>']
        for token in chat_tokens:
            text = text.replace(token, '')
        
        # Remove common unwanted phrases and feedback
        unwanted_phrases = [
            "Here's the complete function code:",
            "Here is the complete function:",
            "The complete function is:",
            "Here's the function:",
            "Complete function:",
            "# Test the function",
            "# Testing the function:",
            ">>> task_func(",
            "Your code looks correct",
            "Your answer is correct",
            "Good job!",
            "Well done!",
            "This code",
            "The code above",
            "It calculates",
            "It uses the given requirements",
        ]
        for phrase in unwanted_phrases:
            text = text.replace(phrase, '')
        
        # Split on common separators and take the first part (the actual function)
        text = text.split('# Test')[0]  # Remove test sections
        text = text.split('>>>')[0]      # Remove interactive examples
        text = text.split('<end_of_turn>')[0]  # Remove anything after Gemma end token
        text = text.split('</end_of_turn>')[0]  # Also handle incorrect closing format
        
        # Remove any text that looks like feedback after code
        lines = text.split('\n')
        code_lines = []
        code_started = False
        
        for line in lines:
            # If we see a function definition, we're in code
            if line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('from '):
                code_started = True
            
            # If we're in code and see feedback-like text, stop
            if code_started and any(feedback in line.lower() for feedback in 
                                  ['your code', 'correct', 'good job', 'well done', 'this code', 'the code']):
                break
                
            code_lines.append(line)
        
        text = '\n'.join(code_lines)
        
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # If no code block, return cleaned text directly
            return text.strip()
            
    def run_code_generation_test(
            self, 
            prompt_helper: Callable, 
            num_tests: int, 
            output_file_path: str,
            prompt_type: str,
            task_set: str,
            continue_from_task: str = None,
            mutations: List[str] = None,
            model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
        ) -> int:
        
        if prompt_type == PromptTypes.ONE_SHOT:
            example_helper= OpenEndedPromptTemplate.structure_one_shot_example
        elif prompt_type == PromptTypes.FEW_SHOT:
            example_helper = OpenEndedPromptTemplate.structure_few_shot_examples
        else:
            example_helper = None
        
        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('o')[-1])
        else:
            continue_from = 0

        num_tests = min(self.question_database.count_documents({}) - continue_from, num_tests)         # ensuring that the number of iterations is lower than max number of documents in the db
        
        if not check_for_mutation_conflicts(mutations=mutations):
            raise ValueError("An invalid combination of mutations were used.")

        for mutation in mutations:
            if mutation not in CodeGeneration.MUTATIONS or mutation == LexicalMutations.LITERAL_FORMAT:
                raise ValueError(f"{mutation} mutation is an invalid mutation for code generation.")
            
        if task_set not in CodeGeneration.BENCHMARKS:
            raise ValueError(f"{task_set} is an invalid benchmark dataset for code generation. Only {CodeGeneration.BENCHMARKS} datasets are valid.")
        
        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        failed_validity = []            # list storing the test case id that have failed the check functions
        timeout = 8                     # int variable indicating the number of seconds the LLM generated program should complete running by
        using_GPU = True if (torch.cuda.is_available() or CodeGenerationTester.is_colab()) else False

        if using_GPU:
            llm = TransformersCodeLLM(model_name=model_name)
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
                task_id = f"{task_set}o{idx}"

                qn_sample = self.question_database.find_one({"_id": task_id})

                if qn_sample is None:                                           # skip to the next task if unable to extract the specific qn id from MongoDB
                    continue

                prompt_template = prompt_helper()

                # storing all the file names in the current directory as a snapshot
                # this is a necessary step to remove any new files created from running the tasks
                curr_dir = os.getcwd()
                snapshot_dir = os.listdir(curr_dir)
            
                qn = qn_sample['qn']                                            # contains the question without the docstring description
                qn_desc = qn_sample['qn_desc']                                  # doc string description
                examples = qn_sample['examples']                                # dict object containing all examples pertaining the question for few shot/one shot prompting
                test_function = qn_sample['check']                              # check function for testing validity of a solution
                canon_soln = qn_sample['canon_solution']                        # canonical solution to the question
                complete_soln = qn + '\n' + canon_soln                          # complete working solution combined from the qn and canon solution

                # Dictionary storing all relevant log data
                log_data_entry = {
                    "task_id": task_id,
                    "prompt" : None,
                    "model_output": None,
                    "check_function": None,
                    "canonical_solution": complete_soln,
                    "failure_type": None
                }
                
                match task_set:
                    case "HumanEval":
                        test_set_helper = CodeGenerationHumanEvalHelper
                    case "BigCodeBench":
                        test_set_helper = CodeGenerationBigCodeBenchHelper
                    case _:
                        log_data_entry['failure_type'] = "invalid_task_set"
                        CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data=log_data_entry)
                        continue

                # Sanity check to filter out test cases where there is only 1 example and hence the task cannot be used for few shot
                if prompt_type == "few_shot" and len(examples.keys()) == 1:
                    print(f"Skipping {task_id} as the complete solution does not have more than 1 example.")
                    log_data_entry['failure_type'] = "insufficient_few_shot_examples"
                    CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data=log_data_entry)
                    continue

                if task_set == "HumanEval":
                    # Obtaining the function name of the task function
                    func_name = qn_sample['func_name']
                else:
                    func_name = 'task_func'

                # Sanity check to ensure that the complete solution passes the check functions
                check_soln_validity = test_set_helper.check_test_case(
                    test_case = test_function, 
                    code_snippet = complete_soln, 
                    func_name = func_name
                    )
                
                if check_soln_validity is not True:
                    failed_validity.append(task_id)
                    print(f"Skipping {task_id} as the complete solution did not pass the check function.")
                    log_data_entry['failure_type'] = "canonical_sol_did_not_pass_check"
                    CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data=log_data_entry)
                    continue
                
                # Handling Task Mutation (If any)
                codemutator = CodeMutator(
                    func_name=func_name,
                    mutated_dict = {
                        'question' : qn,
                        'qn_desc': qn_desc,
                        'examples' : examples,
                        'check_function' : test_function,
                        'full_sol' : complete_soln
                    },
                    benchmark_set=task_set
                )

                for mutation in mutations:
                    codemutator.mutate_for_code_generation(
                        mutation_type=mutation,
                        task_set=task_set
                    )
                    func_name = codemutator.func_name

                # updating log_data_entry with mutated programs (if any)
                log_data_entry['canonical_solution'] = codemutator.mutated_dict['full_sol']
                log_data_entry['check_function'] = codemutator.mutated_dict['check_function']

                # Formating of examples into doc test format for one shot/few shot prompts
                if example_helper is not None:
                    prompt_examples = example_helper(codemutator.mutated_dict['examples'])

                input_variables = {
                    'code': codemutator.mutated_dict['question'],
                    'task': codemutator.mutated_dict['qn_desc'],
                    'example': prompt_examples if example_helper is not None else None,
                }

                log_data_entry["prompt"] = prompt_template.format(**input_variables)

                if using_GPU:
                    ans_dict = llm.invoke(
                        input_variables=input_variables,
                        prompt_template=prompt_template
                    )

                    ans = ans_dict['ans']
                    try:
                        ans = CodeGenerationTester.process_llm_ans(ans)
                    except ValueError:
                        pass

                    log_data_entry['model_output'] = (ans, type(ans))                                            # storing model answer into the database entry
                else:

                    try: 
                        # Running the llm on the input variables and the prompt template
                        ans = self.execute_llm(input_variables = input_variables, prompt_template = prompt_template, llm_model = llm, model_name=model_name)

                    except Exception as e:
                        log_data_entry["failure_type"] = (f"{LLMExecutionRuntimeError.__name__} > {e}")
                        CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                        continue
                    
                    try:
                        # Processing of the llm answer. Some llm answers are in Python code blocks, which needs to be processed as it will fail exec()
                        ans = CodeGenerationTester.process_llm_ans(ans)
                    
                    except ValueError:                          # Raised when the llm answer did not have a python code block
                        try: 
                            exec(ans)                           # Attempting to run the llm answer directly. In some cases, the returned answer can be directly run as no code block was returned
                        except Exception as e:                  # Else, if the answer is not in a valid code block and cannot be run directly, it is a faulty answer and is stored accordingly.
                            print(f"Could not process LLM answer: {e}")
                            log_data_entry["model_output"] = ans
                            log_data_entry["failure_type"] = ("could_not_parse_LLM_answer", type(e))
                            CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                            continue
                    

                log_data_entry['model_output'] = ans       # storing the answer in input_data dict
                ## LLM Answer Test Execution
                try:
                    # multiprocessing library is used here as some LLM answers are wrong and uses a while loop which runs indefinitely.
                    #   This ensures that the LLM answer execution will automatically timeout after timeout seconds
                    multiprocessing_queue = multiprocessing.Queue()

                    verify_answer_process = multiprocessing.Process(        
                        target= test_set_helper.run_llm_answer,
                        args = (ans, codemutator.mutated_dict['check_function'], func_name, multiprocessing_queue)
                        )

                    verify_answer_process.start()
                    verify_answer_process.join(timeout=timeout)

                    over_run = False
                    if verify_answer_process.is_alive():
                        verify_answer_process.kill()
                        verify_answer_process.join()
                        over_run = True
                    
                    curr_dir_snapshot = os.listdir(curr_dir)
                    for file_name in curr_dir_snapshot:
                        if file_name not in snapshot_dir:
                            file_path = os.path.join(curr_dir, file_name)
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            else:
                                os.remove(file_path)
                                
                    if not multiprocessing_queue.empty():
                        error = multiprocessing_queue.get()
                        raise AssertionError(error)
                    elif over_run:
                        raise AssertionError(RuntimeError)
                    
                    multiprocessing_queue.close()
                    multiprocessing_queue.join_thread()
                    
                    task_pass_count += 1

                except Exception as e:
                    log_data_entry['failure_type'] = type(e)            # Logging failure type into log_data_entry

                # logging completed run into csv 
                CodeGenerationTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                
            return task_pass_count
        except Exception as e:
            print(e)
            print(task_id)
            return task_pass_count
        
        except KeyboardInterrupt:
            print(task_id)
            return task_pass_count

class LLMExecutionError(Exception):
    def __init__(self, error):
        super().__init__(error)

class LLMExecutionRuntimeError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


if __name__ == "__main__":
    llm_tester = CodeGenerationTester("HumanEval_Open_Ended")
