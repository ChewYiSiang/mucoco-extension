import pandas as pd
from typing import Tuple, Any, List
import ast

class DataLogHelper:
    @staticmethod
    def check_valid_failure(failure: str):
        if isinstance(failure, float) or (isinstance(failure, str) and "AssertionError" in failure and "Mutation" not in failure):
            return True
        return False

    @staticmethod
    def compare_code_generation_dataframe_results(log1: pd.DataFrame, log2: pd.DataFrame) -> Tuple[int, int]:
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
        # Copying the input logs
        log1_orig, log2_orig = log1.copy(), log2.copy()

        # ## Checking that the log1 column names are equal to log2 column names
        # if not log1.columns.equals(log2.columns):
        #     raise ValueError("CSV column headers do not match.")
        
        ## Checking that both logs have the same number of entries
        # if log1.shape[0] != log2.shape[0]:
        #     raise ValueError("Dataframe shapes are not equal. Double check the entries again.")

        ## If either logs are empty, (0,0) is returned
        if log1.shape[0] == 0 or log2.shape[0] == 0:
            return 0, 0
        
        log1_inconsistencies = 0        # inconsistencies from log1
        log2_inconsistencies = 0        # inconsistencies from log2
        tot = 0                         # union between tasks solved correctly in both logs
        log1_total_answered = 0
        log2_total_answered = 0
        both_failed = 0                 # tasks where both logs failed
        identical_mutation_errors = 0   # tasks with IdenticalMutationError
        both_succeeded = 0              # tasks where both logs succeeded
        count = 0

        total_tasks = log1.shape[0]
        # print(f"Starting comparison of {total_tasks} tasks...")

        ## Checking for inconsistencies between both logs
        for idx in range(total_tasks):
            log1_data = log1.loc[idx]
            log1 = log1.drop(index = idx)

            task_id = log1_data['task_id']
            log_2_matched_data = log2[log2["task_id"] == task_id]

            if log_2_matched_data.shape[0] != 1:
                # raise  ValueError(f"Expected exactly one matched task_id in log_2, but found {log_2_matched_data.shape[0]} matched task_id.")
                continue

            log2_data = log_2_matched_data.iloc[0]
            log2_data_index = log_2_matched_data.index[0]
            log2 = log2.drop(index = log2_data_index)

            log1_result = log1_data['failure_type']
            log2_result = log2_data['failure_type']

            if isinstance(log1_result, float) or isinstance(log1_result, str) and AssertionError.__name__ in log1_result and "Mutation" not in log1_result:
                log1_total_answered += 1

            if isinstance(log2_result, float) or isinstance(log2_result, str) and AssertionError.__name__ in log2_result and "Mutation" not in log2_result:
                log2_total_answered += 1

            # if DataLogHelper.check_valid_failure(log1_result) and DataLogHelper.check_valid_failure(log2_result) and not ("AssertionError" in str(log1_result) and "AssertionError" in str(log2_result)):
            if (isinstance(log1_result, float) and (isinstance(log2_result, str) and AssertionError.__name__ in log2_result) and "Mutation" not in log2_result) or (
                isinstance(log2_result, float) and (isinstance(log1_result, str) and AssertionError.__name__ in log1_result)) or (
                isinstance(log1_result, float) and isinstance(log2_result, float)):
                tot += 1
                # Check if both succeeded (both are NaN/float)
                if isinstance(log1_result, float) and isinstance(log2_result, float):
                    
                    both_succeeded += 1
                elif (isinstance(log1_result, str) and "AssertionError" in log1_result) and (isinstance(log2_result, str) and "AssertionError" in log2_result):
                    pass
                # One succeeded, one failed - this is an inconsistency
                elif not isinstance(log2_result, float):
                    # print(task_id, log1_result, log2_result)
                    log2_inconsistencies += 1
                elif not isinstance(log1_result, float):
                    log1_inconsistencies +=1
                    # print(task_id, log1_result, log2_result)
        # print(log1_inconsistencies + log2_inconsistencies)
        


        ## Checking if log1 have any remaining entries. This is not used now, but could come in handy in the future.
        # if log1.shape[0] > 0:
        #     for idx in range(log1.shape[0]):
        #         task = log1.loc[idx]
        #         unmatched_ids.add(task["task_id"])
        
        ## Checking if log2 have any remaining entries. This is not used now, but could come in handy in the future.
        # if log2.shape[0] > 0:
        #     for idx in range(log2.shape[0]):
        #         task = log2.loc[idx]
        #         unmatched_ids.add(task["task_id"])

        mask1 = (
            log1_orig['failure_type'].astype(str).str.contains("AssertionError", na=False)
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
            'total_inconsistency_questions': tot,
            'log1_success': log1_total_answered - mask1.sum(),
            'log2_success': log2_total_answered - mask2.sum(),
            'log1_total_answered': log1_total_answered,
            'log2_total_answered': log2_total_answered,
            'total_tasks': total_tasks
        }
