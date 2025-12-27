import pandas as pd
from typing import Tuple, Any, List, Dict
from database import MongoDBHelper
from itertools import combinations
from utility.data_log_functions import DataLogHelper
from utility.constants import CodeGeneration, InputPrediction, OutputPrediction


LLM_EXECUTION_ERROR = "LLM Execution Error"
LLM_CORRECTNESS_ERROR = "LLM Correctness Error"
LLM_ANSWER_CORRECT = "LLM Answer Correct"
MUTATED_TASK = "Mutated"
ORIGINAL_TASK = "Original"

class TurbulenceLogHelper:
    RUNNER_PY = r"""

    """


    def __init__(self, task: str):
        db_helper = MongoDBHelper()
        self.task = task
        self.test_suite_outcomes = {}


        self.turbulence_db = db_helper.client["Baseline_Questions_DB"]["Turbulence_Benchmark"]
        self.total_questions= self.turbulence_db.count_documents({})
        self.total_tasks = 0

        for idx in range(1,self.total_questions+1):
            task_id = f"TurbulenceQ{idx}"
            qn = self.turbulence_db.find_one({"_id":task_id})
            self.total_tasks += len(qn['params']) if qn != None else 0

    def run_candidate_in_subprocess(self, candidate: str, test_suite: str, timeout_s:int = 10) -> Dict:
        pass
    
    def obtain_inconsistency_difference(self, row1: pd.Series, row2: pd.Series) -> Dict:

        model_output1 = row1['model_output']
        model_output2 = row2['model_output']


        inconsistencies = {}

        if self.task == CodeGeneration.NAME:
            check_function = row1['check_function']

            task_id1 = row1['task_id']
            task_id2 = row2['task_id']

            func_input1 = row1['func_input']
            func_input2 = row1['func_input']

            # inconsistency test with test suite
            if not self.test_suite_outcomes.get(task_id1, None):
                test_outcome1 = self.run_candidate_in_subprocess(
                    candidate = model_output1,
                    test_suite = check_function,
                )
            else:
                test_outcome1 = self.test_suite_outcomes[task_id1]

            pass
        else:
            inconsistencies['inconsistency_exists'] = model_output1 != model_output2

        return inconsistencies

    
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

            # pairwise comparisons between entries with the same question template
            for (_, row1), (_, row2) in combinations(log_task_qns.iterrows(), 2):
                failure_type1 = row1['failure_type']
                failure_type2 = row2['failure_type']
                row1_result = DataLogHelper.verify_failure_type_relevance(failure_type1)
                row2_result = DataLogHelper.verify_failure_type_relevance(failure_type2)


                # determining the outcome of the llm output: either correct or incorrect
                res1_correct = row1_result[LLM_ANSWER_CORRECT]
                res2_correct = row2_result[LLM_ANSWER_CORRECT]
                res1_wrong = row1_result[LLM_CORRECTNESS_ERROR]
                res2_wrong = row2_result[LLM_CORRECTNESS_ERROR]
                res1_invalid = row1_result[LLM_EXECUTION_ERROR]
                res2_invalid = row2_result[LLM_EXECUTION_ERROR]

                if res1_correct and res2_correct:
                    pass

                elif res1_invalid and res2_invalid:
                    pass

                elif res1_correct and (res2_invalid or res2_wrong) or res2_correct and (res1_invalid or res1_wrong):
                    inconsistency_count += 1
                    # TODO: check inconsistency distance

                elif res1_wrong and res2_wrong:
                    pass
                    # TODO: check inconsistency distance
                    # TODO: if inconsistency exists: inconsistency count += 1

                elif (res1_invalid and res2_wrong) or (res1_wrong and res2_invalid):
                    inconsistency_count += 1
                    # no need to check inconsistency distance

                else:
                    continue

                total_comparisons += 1
        
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
            num_questions += len(log_task_qns)
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
