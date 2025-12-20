import pandas as pd
from typing import Tuple, Dict, Any
from code_generation.code_generation_tester import LLMExecutionRuntimeError, LLMExecutionError
from utility.constants import CodeGeneration
from prediction_inconsistency.utility.database_helper import extract_assert_cases
class DataLogHelper:
    def compare_model_outputs(model_output1: Any, model_output2: Any, task: str, check_function: str) -> bool:
        if task == CodeGeneration.NAME:
            _, test_cases, _ = extract_assert_cases(check_function)
            print(test_cases)
        else:
            return model_output1 == model_output2

    def inconsistency_scoring(
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
        

        # 1. check if both llm outputs are correct > no inconsistencies
        if isinstance(failure_type1, float) and isinstance(failure_type2, float):
            inconsistencies['inconsistency_comparison'] += 1
        
        # 2. check if either res1 is wrong while res2 is correct
        elif is_valid_str_failure(failure_type1) and isinstance(failure_type2, float):
            inconsistencies['inconsistency_res1'] += 1
            inconsistencies['total_inconsistencies'] += 1
            inconsistencies['inconsistency_comparison'] += 1

        
        # 3. check if either res2 is wrong while res1 is correct
        elif is_valid_str_failure(failure_type2) and isinstance(failure_type1, float):
            inconsistencies['inconsistency_res2'] += 1
            inconsistencies['total_inconsistencies'] += 1
            inconsistencies['inconsistency_comparison'] += 1


        # 4. check if res1 and res2 are both wrong
        elif is_valid_str_failure(failure_type1) and is_valid_str_failure(failure_type2):
            inconsistent_errors = DataLogHelper.compare_model_outputs(
                model_output1=model_output1,
                model_output2= model_output2,
                task = task,
                check_function= check_function
            )
            if inconsistent_errors:
                inconsistencies['total_inconsistencies'] += 1

            inconsistencies['inconsistency_comparison'] += 1

        # 5. else, either of them have some other errors that do not contribute to inconsistency
        else:
            pass

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
            valid_parse_error = "could_not_parse_LLM_answer" not in model_output

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
        # print(f"Starting comparison of {total_tasks} tasks...")

        ## Checking for inconsistencies between both logs
        for idx in range(total_tasks):
            log1_data = log1.loc[idx]
            log1 = log1.drop(index = idx)

            task_id = log1_data['task_id']
            print(task_id)
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
            inconsistency_scores = DataLogHelper.inconsistency_scoring(
                log1_data = log1_data,
                log2_data = log2_data,
                task = task
            )

            log1_inconsistencies += inconsistency_scores['inconsistency_res1']
            log2_inconsistencies += inconsistency_scores['inconsistency_res2']
            total_inconsistencies += inconsistency_scores['total_inconsistencies']
            total_comparisons += inconsistency_scores['inconsistency_comparison']

        mask1 = (
            log1_orig['failure_type'].astype(str).str.contains(AssertionError.__name__, na=False)
            & ~log1_orig['failure_type'].astype(str).str.contains("Mutation", na=False)
        )

        mask2 = (
            log2_orig['failure_type'].astype(str).str.contains("AssertionError", na=False)
            & ~log2_orig['failure_type'].astype(str).str.contains("Mutation", na=False)
        )

        # print(f"\n=== COMPARISON SUMMARY ===")
        # print(f"Total tasks processed: {total_tasks}")
        # print(f"Both succeeded: {both_succeeded}")
        # print(f"Both failed: {both_failed}")
        # print(f"IdenticalMutationError: {identical_mutation_errors}")
        # print(f"Log1 Assertion Errors {mask1.sum()}")
        # print(f"Log2 Assertion Errors {mask2.sum()}")
        # print(f"Comparable tasks (atleast one succeeded): {tot}")
        # print(f"  - Log1 failed, Log2 succeeded: {log1_inconsistencies}")
        # print(f"  - Log1 succeeded, Log2 failed: {log2_inconsistencies}")
        # total_inconsistencies = log1_inconsistencies + log2_inconsistencies
        # print(f"Total inconsistencies: {total_inconsistencies}/{tot}")

        # return f"{log1_inconsistencies}/{tot}", f"{log2_inconsistencies}/{tot}"
        # print('----')
        return {
            'log1_inconsistencies': log1_inconsistencies,
            'log2_inconsistencies': log2_inconsistencies,
            'total_inconsistencies': total_inconsistencies,
            'total_inconsistency_questions': total_comparisons,
            'log1_success': log1_total_answered - mask1.sum(),
            'log2_success': log2_total_answered - mask2.sum(),
            'log1_total_answered': log1_total_answered,
            'log2_total_answered': log2_total_answered,
            'total_tasks': total_tasks
        }


