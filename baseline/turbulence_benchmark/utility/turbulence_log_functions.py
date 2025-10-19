import pandas as pd
from typing import Tuple, Any, List, Dict
from database import MongoDBHelper
from itertools import combinations
from pymongo.mongo_client import MongoClient


class TurbulenceLogHelper:
    def __init__(self):
        db_helper = MongoDBHelper()
        self.turbulence_db = db_helper.client["Baseline_Questions_DB"]["Turbulence_Benchmark"]
        self.total_questions= self.turbulence_db.count_documents({})
        self.total_tasks = 0

        for idx in range(1,self.total_questions+1):
            task_id = f"TurbulenceQ{idx}"
            qn = self.turbulence_db.find_one({"_id":task_id})
            self.total_tasks += len(qn['params']) if qn != None else 0

    def obtain_mucoco_code_inconsistency_score(self, log1: pd.DataFrame, log2: pd.DataFrame) -> Dict[str, int]:
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

        ## Checking that the log1 column names are equal to log2 column names
        if not log1.columns.equals(log2.columns):
            raise ValueError("CSV column headers do not match.")

        ## If either logs are empty, (0,0) is returned
        if log1.shape[0] == 0 or log2.shape[0] == 0:
            return 0, 0
        
        log1_inconsistencies = 0        # inconsistencies from log1
        log2_inconsistencies = 0        # inconsistencies from log2
        tot = 0                         # union between tasks solved correctly in both logs
        both_failed = 0                 # tasks where both logs failed
        both_succeeded = 0              # tasks where both logs succeeded

        # print(f"Starting comparison of {self.total_questions} tasks...")

        ## Checking for inconsistencies between both logs

        for idx in range(1, self.total_questions+1):
            task_id = f"TurbulenceQ{idx}"
            log1_task_qns = log1[log1['task_id'].str.contains(rf'^{task_id}(?=_|$)', regex=True)].reset_index(drop = True)
            log2_task_qns = log2[log2['task_id'].str.contains(rf'^{task_id}(?=_|$)', regex=True)].reset_index(drop = True)

            # log1_task_qns = log1[log1['task_id'] == task_id].reset_index(drop=True)
            # log2_task_qns = log2[log2['task_id'] == task_id].reset_index(drop=True)

            if len(log1_task_qns) != len(log2_task_qns):
                raise ValueError("Both logs do not have the same number of questions.")

            for task_idx in range(len(log1_task_qns)):
                log1_data = log1_task_qns.loc[task_idx]
                log2_data = log2_task_qns.loc[task_idx]

                log1_result = log1_data['failure_type']
                log2_result = log2_data['failure_type']

                if (isinstance(log1_result, float) and (isinstance(log2_result, str) and AssertionError.__name__ in log2_result)) or (
                    isinstance(log2_result, float) and (isinstance(log1_result, str) and AssertionError.__name__ in log1_result)) or (
                    isinstance(log1_result, float) and isinstance(log2_result, float)):
                    tot += 1
                    # Check if both succeeded (both are NaN/float)
                    if isinstance(log1_result, float) and isinstance(log2_result, float):
                        both_succeeded += 1
                    # One succeeded, one failed - this is an inconsistency
                    elif not isinstance(log2_result, float):
                        log2_inconsistencies += 1
                    elif not isinstance(log1_result, float):
                        log1_inconsistencies +=1
                

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

        print(f"\n=== COMPARISON SUMMARY ===")
        print(f"Total tasks processed: {self.total_questions}")
        print(f"Both succeeded: {both_succeeded}")
        print(f"Both failed: {both_failed}")
        print(f"Comparable tasks (atleast one succeeded): {tot}")
        print(f"  - Log1 failed, Log2 succeeded: {log1_inconsistencies}")
        print(f"  - Log1 succeeded, Log2 failed: {log2_inconsistencies}")
        total_inconsistencies = log1_inconsistencies + log2_inconsistencies
        print(f"Total inconsistencies: {total_inconsistencies}/{tot}")

        # return f"{log1_inconsistencies + log2_inconsistencies}/{tot}", round((log1_inconsistencies + log2_inconsistencies)*100/tot,2)
        return {
            "log1_inconsistencies" : log1_inconsistencies,
            "log2_inconsistencies" : log2_inconsistencies,
            "tot" : tot
        }
    
    def obtain_turbulence_code_inconsistency_score(self, log: pd.DataFrame) -> Dict[str, float]:
        """
        This method returns the code inconsistency score of the turbulence benchmark.

        The code inconsistency score of the turbulence benchmark is calculated through pairwise comparisons of question instances of the same template.
        """
        inconsistency_count = 0
        total_comparisons = 0
        log1_assertion = 0
        log1_correct = 0


        for idx in range(1, self.total_questions+2):
            task_id = f"TurbulenceQ{idx}"
            log_task_qns = log[log['task_id'].str.contains(rf'^{task_id}(?:_|$)', regex=True)]
            for idx, l in log_task_qns.iterrows():
                if isinstance(l['failure_type'], float):
                    log1_correct +=1 
                elif "AssertionError" in l['failure_type'] and "Mutation" not in l['failure_type']:
                    log1_assertion += 1

            for (_, row1), (_, row2) in combinations(log_task_qns.iterrows(), 2):
                row1_failure_type = str(row1['failure_type']).strip()
                row2_failure_type = str(row2['failure_type']).strip()

                if not any(
                    (isinstance(f, float) or (isinstance(f, str) and "assertionerror" in f.lower() and "mutation" not in f.lower()))
                    for f in [row1_failure_type, row2_failure_type]
                ) and not ("AssertionError" in row1_failure_type and "AssertionError" in row2_failure_type):
                    continue
                total_comparisons += 1
                if row1_failure_type != row2_failure_type:
                
                    # print(task_id, idx1, row1['failure_type'], idx2, row2['failure_type'])
                    inconsistency_count += 1
        
        # return f"{inconsistency_count}/{total_comparisons}", f"{round(inconsistency_count*100/total_comparisons, 2)}"
        return {
            "inconsistency_count": inconsistency_count,
            "total_comparisons": total_comparisons,
            "correct_instances" : log1_correct,
            "incorrect_instances": log1_assertion
        }

    def obtain_question_inconsistency_count(self, log: pd.DataFrame) -> Dict[str, float]:
        """
        This method obtains the number of inconsistent questions in the Turbulence dataset.
        These score measures inconsistency between question instances of the same template. 
        
        For example, if 10 tasks are instantiated from a template and 1 of the tasks was incorrect, this question is considered "inconsistent" at a question template level.

        Args: 
            log (pd.Dataframe): Pandas dataframe of the log results

        Returns:
            Dict[str, float]: A dictionary containing the question inconsistency count and total number of tasks
        """
        inconsistent_qn_count = 0
        valid_qn_count = 0
        num_questions = 0

        for idx in range(1, self.total_questions+2):
            task_id = f"TurbulenceQ{idx}"
            log_task_qns = log[log['task_id'].str.contains(rf'^{task_id}(?:_|$)', regex=True)].reset_index(drop = True)

            if not any(
                (isinstance(f['failure_type'], float) or (isinstance(f['failure_type'], str) and "assertionerror" in f.to_string().lower() and "mutation" not in f.to_string().lower()))
                for idx, f in log_task_qns.iterrows()
            ):
                continue
            
            valid_qn_count += 1
            num_questions += len(self.turbulence_db.find_one({"_id": task_id})['params'])
            consistency_type = None
            for task_idx in range(len(log_task_qns)):
                failure_type = log_task_qns.loc[task_idx]['failure_type']
                                
                if consistency_type is None:
                    consistency_type = str(failure_type).strip()
                elif consistency_type != str(failure_type).strip():
                    inconsistent_qn_count += 1
                    break
                
                    
        # return f"{inconsistent_qn_count}/{self.total_questions}", f"{round(inconsistent_qn_count*100/self.total_questions, 2)}"
        return {
            "inconsistent_qn_count": inconsistent_qn_count,
            "total_questions": valid_qn_count,
            "num_questions": num_questions
        }
