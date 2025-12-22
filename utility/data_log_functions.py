import re
import builtins
import pandas as pd
from typing import Tuple, Dict, Any, List
from code_generation.code_generation_tester import LLMExecutionRuntimeError, LLMExecutionError
from utility.constants import CodeGeneration
from prediction_inconsistency.utility.database_helper import extract_assert_cases
import multiprocessing as mp
from utility.custom_decorators import multiprocessing_method
import signal

class TimeoutError(Exception): pass

# @multiprocessing_method
# def run_tests(
#         func_name: str, 
#         test_inputs: List[Any],
#         test_outputs:  List[Any],
#         llm_solution: str, 
#         results_queue: mp.Queue, 
#     ) -> None:
#     env = {}
#     results = {}

#     for idx in range(len(test_inputs)):
#         results[idx] = False
#     try:
#         exec(llm_solution, env)
#     except Exception as e:
#         results_queue.put(results)
    
#     if isinstance(llm_solution, float):
#         results_queue.put(results)

#     for idx, (test_data, test_output) in enumerate(zip(test_inputs, test_outputs)):            
#         test_input, test_metadata = test_data

#         try:
#             if isinstance(test_metadata, list):
#                 result = env[func_name](*test_input)
#             else:
#                 result = env[func_name](test_input)
            
#             results[idx] = test_output == result
#         except Exception:
#             results[idx] = False

#     results_queue.put(results)
    
class HumanEvalHelper:
    def extract_function_name(prompt: str) -> str:
        """
        Extract the user-defined function being tested in doctest examples,
        excluding Python built-ins (e.g., round, len).
        """
        builtin_names = set(dir(builtins))

        # Only look at doctest lines
        doctest_lines = [
            line for line in prompt.splitlines()
            if line.strip().startswith(">>>")
        ]

        for line in doctest_lines:
            # Find ALL function calls in the line
            calls = re.findall(r"([a-zA-Z_]\w*)\s*\(", line)
            for name in calls:
                if name not in builtin_names:
                    return name

        raise ValueError("No non-builtin function call found in doctest examples")

    def humaneval32_inconsistency_score(prompt:str, candidate: str) -> int:
        def poly(xs: list, x: float):
            """
            Evaluates polynomial with coefficients xs at point x.
            return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
            """
            import math
            return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

        def check(candidate):
            import math
            import random
            rng = random.Random(42)
            import copy
            pass_count = 0
            for _ in range(100):
                ncoeff = 2 * rng.randint(1, 4)
                coeffs = []
                for _ in range(ncoeff):
                    coeff = rng.randint(-10, 10)
                    if coeff == 0:
                        coeff = 1
                    coeffs.append(coeff)
                solution = candidate(copy.deepcopy(coeffs))
                try:
                    assert math.fabs(poly(coeffs, solution)) < 1e-4
                    pass_count += 1
                except AssertionError:
                    return pass_count
            return pass_count
        env = {}
        exec(candidate, env)
        candidate_func_name = HumanEvalHelper.extract_function_name(prompt)

        return check(candidate=env[candidate_func_name])
    
    # def humaneval_inconsistency_score(prompt: str, candidate: str | float, test_inputs: List[Any], test_outputs: List[Any]) -> Dict[int, int]:
    #     candidate_func_name = HumanEvalHelper.extract_function_name(prompt)

    #     TIMEOUT = 10
    #     results_queue = mp.Queue()

    #     run_tests_process = mp.Process(
    #         target= run_tests,
    #         kwargs={
    #             "func_name": candidate_func_name,
    #             "test_inputs": test_inputs,
    #             "test_outputs": test_outputs,
    #             "llm_solution": candidate,
    #             "results_queue": results_queue
    #         }
    #     )

    #     run_tests_process.start()
    #     run_tests_process.join(timeout=TIMEOUT)

    #     if run_tests_process.is_alive():
    #         run_tests_process.kill()
    #         run_tests_process.join()
    #         raise RuntimeError("Ran for too long.")
    
    #     results = results_queue.get()

    #     return results
    
    def humaneval_inconsistency_score(prompt: str, candidate: str | float, test_inputs: List[Any], test_outputs: List[Any]) -> Dict[int, int]:
        env = {}
        results = {}
        for idx in range(len(test_inputs)):
            results[idx] = False

        if isinstance(candidate, float):
            return results

        candidate_func_name = HumanEvalHelper.extract_function_name(prompt)
        try:
            exec(candidate, env)
        except Exception:
            return results
        
        fn = env[candidate_func_name]

        for idx, (test_data, test_output) in enumerate(zip(test_inputs, test_outputs)):            
            test_input, test_metadata = test_data

            try:
                with time_limit(2):
                    if isinstance(test_metadata, list):
                        result = fn(*test_input)
                    else:
                        result = fn(test_input)
                
                results[idx] = test_output == result
            except Exception:
                results[idx] = False
        return results

class DataLogHelper:
    def compare_model_outputs(
        log1_data: Dict[str, Any],
        log2_data: Dict[str, Any],
        task: str, 
    ) -> Dict[str, int | bool]:
        
        print(log2_data['task_id'])

        # Declaring variables
        check_function = log1_data['check_function']
        model_output1 = log1_data['model_output']
        model_output2 = log2_data['model_output']

        task_id = log1_data['task_id']

        prompt_1 = log1_data['prompt']
        prompt_2 = log2_data['prompt']

        inconsistency_dict = {}
        
        # calculating inconsistency metrics
        if task == CodeGeneration.NAME:
            if task_id == "HumanEvalo32":
                inconsistency_score1 = HumanEvalHelper.humaneval32_inconsistency_score(prompt_1, model_output1)
                inconsistency_score2 = HumanEvalHelper.humaneval32_inconsistency_score(prompt_2, model_output2)
                
                inconsistency_dict['inconsistency_exists'] = inconsistency_score1 == inconsistency_score2
                #TODO: fix here

                inconsistency_dict['inconsistency_score'] = {'score1': inconsistency_score1, 'score2': inconsistency_score2}
            else:
                _, test_cases, _ = extract_assert_cases(check_function)
                input_data = []
                output_data = []
                for inputs, output in test_cases:
                    output_data.append(output)
                    input_data.append(inputs)

                print('ok1')

                inconsistency_results1 = HumanEvalHelper.humaneval_inconsistency_score(
                    prompt = prompt_1, 
                    candidate = model_output1, 
                    test_inputs = input_data, 
                    test_outputs = output_data
                )
                
                print('ok')
                inconsistency_results2 = HumanEvalHelper.humaneval_inconsistency_score(
                    prompt = prompt_2, 
                    candidate = model_output2, 
                    test_inputs = input_data, 
                    test_outputs = output_data
                )

                if inconsistency_results1.keys() != inconsistency_results2.keys():
                    raise ValueError("Something went wrong with the check functions as there is a mismatch in the number of input/output test cases.")
                
                inconsistency_score = 0
                for key in inconsistency_results1.keys():
                    res1 = inconsistency_results1[key]
                    res2 = inconsistency_results2[key]
                    inconsistency_score += 1 if res1 != res2 else 0

                inconsistency_dict['inconsistency_exists'] = inconsistency_results1 != inconsistency_results2
                inconsistency_dict['inconsistency_score'] = inconsistency_score/ len(inconsistency_results1)
                    
        else:
            inconsistency_dict['inconsistency_exists'] = model_output1 == model_output2

        return inconsistency_dict

    def inconsistency_heuristics(
            task: str, 
            log1_data: Dict[str, Any],
            log2_data: Dict[str, Any],
            ) -> Dict[str, str]:
        """
        This function applies the heuristics for inconsistency scoring on 2 result inputs.
        The inconsistency from res1, inconsistency from res2 and total inconsistencies

        Dictionary keys and their possible values:
        1. inconsistency_res1: 1 when res1 is wrong while res2 is correct, else 0
        2. inconsistency_res2: 1 when res2 is wrong while res1 is correct, else 0
        3. total_inconsistencies: 1 when res1 or res2 have any inconsistency (such as both assertion error), else 0
        4. inconsistency_comparison: 1 when res1 and res2 are valid outputs for inconsistency comparisons, else 0 

        Args:
            - log1_data: dictionary containing llm output data 
            - log2_data: dictionary containing llm output data 
            - task: task type (e.g.: code_generation etc)
        Returns:
            - dictionary containing inconsistencies scores
        """
        def is_valid_str_failure(res: str | float) -> bool:
            if isinstance(res, str):
                has_assertion_error = AssertionError.__name__ in res and "Mutation" not in res
                has_llm_runtime_error = (LLMExecutionRuntimeError.__name__ in res) or (LLMExecutionError.__name__ in res)
                failed_to_parse_llm_ans = "could_not_parse_LLM_answer" in res

                if has_assertion_error or has_llm_runtime_error or failed_to_parse_llm_ans:
                    return True
                return False
            return False
        
        inconsistencies = {
            "inconsistency_res1": 0,
            "inconsistency_res2": 0,
            "total_inconsistencies": 0,
            "inconsistency_comparison": 0
        }

        # Extracting key variables from log data
        failure_type1 = log1_data['failure_type']
        failure_type2 = log2_data['failure_type']

        # determining the outcome of the llm output: either correct or incorrect
        res1_correct = isinstance(failure_type1, float)
        res2_correct = isinstance(failure_type2, float)
        res1_wrong = is_valid_str_failure(failure_type1)
        res2_wrong = is_valid_str_failure(failure_type2)

        def run_comparison():
            return DataLogHelper.compare_model_outputs(
                log1_data=log1_data,
                log2_data=log2_data,
                task=task,
            )

        # 1. check if both llm outputs are correct > no inconsistencies
        if res1_correct and res2_correct:
            inconsistency_errors = {}

        # 2. check if either res1 is wrong while res2 is correct
        elif res1_wrong and res2_correct:
            inconsistencies['inconsistency_res1'] += 1
            inconsistency_errors = run_comparison()

        # 3. check if either res2 is wrong while res1 is correct
        elif res2_wrong and res1_correct:
            inconsistencies['inconsistency_res2'] += 1
            inconsistency_errors = run_comparison()

        # 4. check if res1 and res2 are both wrong
        elif res1_wrong and res2_wrong:
            inconsistency_errors = run_comparison()

        # 5. else, either of them have some other errors that do not contribute to inconsistency
        else:
            return inconsistencies
        
        inconsistencies['inconsistency_comparison'] += 1
        if inconsistency_errors.get('inconsistency_exists', False):
            inconsistencies['total_inconsistencies'] += 1

        return inconsistencies

    @staticmethod
    def compare_code_generation_dataframe_results(log1: pd.DataFrame, log2: pd.DataFrame, task: str) -> Tuple[int, int]:
        """
        This function is used to compare between two pd dataframes containing the logs of two comparable code generation runs and returns any inconsistencies found between the two logs.
        
        A sample use case is as follows:

            ``` python
            log1_file_path = proj_dir + "/results/mistral-small-2506_zero_shot_no_mutation.csv"
            log2_file_path = proj_dir + "/results/mistral-small-2506_zero_shot_random.csv"

            log1 = pd.read_csv(log1_file_path)
            log2 = pd.read_csv(log2_file_path)

            log1_inconsistencies, log2_inconsistencies = compare_code_generation_dataframe_results(log1=log1, log2=log2)
            ```
        
        log1_inconsistencies refers to code inconsistencies where the same task was solved correctly in log2 but was solved incorrectly in log1.
        On the other hand, log2_inconsistencies refers to code inconsistencies where the same task was solved correctly in log1 but was solved incorrectly in log2.

        This function assumes that the column names "task_id" and "failure_type" are used in both logs.

        Args:
            log1 (pd.DataFrame): first log entry
            log2 (pd.DataFrame): second log entry
        
        Returns:
            Tuple[int, int]: tuple containing the inconsistencies found in log 1, and inconsistencies found in log 2

        Raises:
            ValueError: Raised when the csv column headers do not match, which indicates different type of logs are being compared OR when the dataframes do not contain the same number of rows

        Potential Improvements:
            This function assumes that the logs contain information from all runs, hence both the dataframe logs MUST have the same number of entries. 
                - Should there be any changes in the future where only selected runs are logged, this function may need to be modifid accordingly. 
    
        """
        def check_llm_output_validity(model_output: str | float) -> bool:
            if isinstance(model_output, float):
                return True

            has_assertion = AssertionError.__name__ in model_output and "Mutation" not in model_output
            valid_parse_error = "could_not_parse_LLM_answer" in model_output

            return has_assertion or valid_parse_error

        # Copying the input logs
        log1_orig, log2_orig = log1.copy(), log2.copy()

        ## If either logs are empty, (0,0) is returned
        if log1.shape[0] == 0 or log2.shape[0] == 0:
            return 0, 0
        
        log1_inconsistencies = 0        # inconsistencies from log1
        log2_inconsistencies = 0        # inconsistencies from log2
        total_comparisons = 0           
        log1_total_answered = 0
        log2_total_answered = 0
        total_inconsistencies = 0

        total_tasks = log1.shape[0]

        ## Checking for inconsistencies between both logs
        for idx in range(total_tasks):
            log1_data = log1.loc[idx]
            log1 = log1.drop(index = idx)

            task_id = log1_data['task_id']
            log_2_matched_data = log2[log2["task_id"] == task_id]

            if log_2_matched_data.shape[0] != 1:
                # raise ValueError(f"Expected exactly one matched task_id in log_2, but found {log_2_matched_data.shape[0]} matched task_id.")
                continue

            log2_data = log_2_matched_data.iloc[0]
            log2_data_index = log_2_matched_data.index[0]
            log2 = log2.drop(index = log2_data_index)

            # checking if the llm answered appropriately
            if check_llm_output_validity(log1_data['failure_type']):
                log1_total_answered += 1

            if check_llm_output_validity(log2_data['failure_type']):
                log2_total_answered += 1
            
            # **Inconsistency Scoring Heuristics**
            inconsistency_scores = DataLogHelper.inconsistency_heuristics(
                log1_data = log1_data,
                log2_data = log2_data,
                task = task
            )

            log1_inconsistencies += inconsistency_scores['inconsistency_res1']
            log2_inconsistencies += inconsistency_scores['inconsistency_res2']
            total_inconsistencies += inconsistency_scores['total_inconsistencies']
            total_comparisons += inconsistency_scores['inconsistency_comparison']

        mask1 = log1_orig['failure_type'].map(type).eq(float)
        mask2 = log2_orig['failure_type'].map(type).eq(float)

        return {
            'log1_inconsistencies': log1_inconsistencies,
            'log2_inconsistencies': log2_inconsistencies,
            'total_inconsistencies': total_inconsistencies,
            'total_inconsistency_comparisons': total_comparisons,
            'log1_success': int(mask1.sum()),
            'log2_success': int(mask2.sum()),
            'log1_total_answered': log1_total_answered,
            'log2_total_answered': log2_total_answered,
            'total_tasks': total_tasks
        }


