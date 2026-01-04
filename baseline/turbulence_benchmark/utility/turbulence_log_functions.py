import pandas as pd
from typing import Tuple, Any, List, Dict
from database import MongoDBHelper
from itertools import combinations
from utility.data_log_functions import DataLogHelper
from utility.constants import CodeGeneration, InputPrediction, OutputPrediction
import subprocess
import sys
import json
import textwrap
import math


LLM_EXECUTION_ERROR = "LLM Execution Error"
LLM_CORRECTNESS_ERROR = "LLM Correctness Error"
LLM_ANSWER_CORRECT = "LLM Answer Correct"
MUTATED_TASK = "Mutated"
ORIGINAL_TASK = "Original"

class TurbulenceLogHelper:
    RUNNER_PY = r"""
    import json
    import sys
    import traceback
    from io import StringIO
    import contextlib

    def main(candidate: str, test_suite: str, test_func_name: str):
        payload = {
            "status": "ok",
            "test_suite_outcome": None,
            "captured_stdout": "",
            "captured_stderr": "",
            "traceback": None,
        }

        # Use a single env so candidate + tests share definitions
        env = {"__builtins__": __builtins__}

        out_buf = StringIO()
        err_buf = StringIO()

        try:
            # Capture *everything* candidate/tests print or log to stderr
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                exec(candidate, env, env)
                exec(test_suite, env, env)

                fn = env.get(test_func_name)
                if fn is None:
                    raise NameError(f"Test function '{test_func_name}' not found in env")
                fn()

            payload["test_suite_outcome"] = "Passed"

        except AssertionError:
            payload["test_suite_outcome"] = "Failed"

        except Exception:
            payload["test_suite_outcome"] = "Error"
            payload["status"] = "error"
            payload["traceback"] = traceback.format_exc()

        finally:
            # Always include what was printed, even if it failed
            payload["captured_stdout"] = out_buf.getvalue()
            payload["captured_stderr"] = err_buf.getvalue()

        # IMPORTANT: exactly ONE print (and it's valid JSON)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()

    if __name__ == "__main__":
        job = json.loads(sys.stdin.read() or "{}")
        main(
            candidate=job["candidate"],
            test_suite=job["test_suite"],
            test_func_name=job["test_func_name"],
        )

    """


    def __init__(self, task: str):
        self.task = task
        self.test_suite_outcomes = {}
        self.test_func_names = {}
        self.failure_types = {}
        self.total_questions = 59

    def run_candidate_in_subprocess(
            self,
            candidate: str, 
            test_suite: str, 
            test_names: List[str], 
            timeout_s:int = 10
        ) -> Dict:

        test_func_outcomes = {t : False for t in test_names}


        for test_name in test_names:
            p = subprocess.Popen(
                [sys.executable, "-c", textwrap.dedent(TurbulenceLogHelper.RUNNER_PY)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            job = {
                "candidate": candidate,
                "test_suite": test_suite,
                "test_func_name": test_name,
            }
            try:
                stdout, stderr = p.communicate(input=json.dumps(job), timeout=timeout_s)
            except subprocess.TimeoutExpired:
                print('expired')
                p.kill()
                test_func_outcomes[test_name] = False
                continue

            if p.returncode != 0:
                print(stderr)
                print('crash')
                # runner crashed (syntax error in RUNNER_PY etc.)
                test_func_outcomes[test_name] = False
                continue

            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                print('bad json')
                test_func_outcomes[test_name] = False
                continue
            
            test_suite_outcome = payload.get("test_suite_outcome")
            test_func_outcomes[test_name] = test_suite_outcome == "Passed"
                
        return test_func_outcomes
    
    def obtain_inconsistency_difference(self, row1: pd.Series, row2: pd.Series) -> Dict:
        """
        obtain the inconsistency details between 2 LLM outputs.
        """

        def obtain_test_func_names(test_suite: str) -> List[str]:
            import ast

            func_names= []
            t = ast.parse(test_suite)
            for n in t.body:
                if isinstance(n, ast.FunctionDef):
                    func_names.append(n.name)
            
            return func_names
        
        model_output1 = row1['model_output']
        model_output2 = row2['model_output']

        inconsistencies = {}

        if self.task == CodeGeneration.NAME:
            check_function = row1['check_function']

            task_id1 = row1['task_id']
            question_template_id = task_id1.split("_")[0]
            
            # Checking if the test function names have already been extracted previously
            if self.test_func_names.get(question_template_id, None):
                test_func_names = self.test_func_names[question_template_id]
            else:
                test_func_names = obtain_test_func_names(test_suite=check_function)
                self.test_func_names[question_template_id] = test_func_names
            
            # inconsistency test with test suite
            def get_test_outcome(data: pd.Series):
                task_id = data['task_id']
                candidate = data['model_output']
                test_suite = data['check_function']
    
                if task_id not in self.test_suite_outcomes:
                    self.test_suite_outcomes[task_id] = self.run_candidate_in_subprocess(
                        candidate = candidate,
                        test_suite=test_suite,
                        test_names = self.test_func_names[question_template_id],
                    )
                return self.test_suite_outcomes[task_id]

            test_outcome1 = get_test_outcome(row1)
            test_outcome2 = get_test_outcome(row2)


            # obtaining inconsistency count
            total_inconsistencies = 0
            for k in test_outcome1:
                t1 = test_outcome1[k]
                t2 = test_outcome2.get(k, None)

                if t1 != t2:
                   total_inconsistencies += 1


            inconsistencies['inconsistency_exists'] = test_outcome1 != test_outcome2
            inconsistencies['inconsistency_distance'] = total_inconsistencies / len(test_outcome1)

        else:
            inconsistencies['inconsistency_exists'] = model_output1 != model_output2

        return inconsistencies

    
    def obtain_turbulence_code_inconsistency_score(self, log: pd.DataFrame) -> Dict[str, float]:
        """
        This method returns the code inconsistency score of the turbulence benchmark.

        The code inconsistency score of the turbulence benchmark is calculated through pairwise comparisons of question instances of the same template.
        """
        inconsistency_count = 0
        inconsistency_distance = 0
        total_comparisons = 0
        total_correct = 0
        total_assertion = 0
        total_invalid = 0


        for idx in range(1, self.total_questions+2):
            task_id = f"TurbulenceQ{idx}"
            log_task_qns = log[log['task_id'].str.contains(rf'^{task_id}(?:_|$)', regex=True)]
            for idx, l in log_task_qns.iterrows():
                qn_id = l['task_id']
                failure_type = l['failure_type']


                result = DataLogHelper.verify_failure_type_relevance(failure_type)
                total_correct += 1 if result[LLM_ANSWER_CORRECT] else 0
                total_assertion += 1 if result[LLM_CORRECTNESS_ERROR] else 0
                total_invalid += 1 if result[LLM_EXECUTION_ERROR] else 0
            
                self.failure_types[qn_id] = result

            # pairwise comparisons between entries with the same question template
            for (_, row1), (_, row2) in combinations(log_task_qns.iterrows(), 2):
                qn_id1 = row1['task_id']
                qn_id2 = row2['task_id']

                row1_result = self.failure_types[qn_id1]
                row2_result = self.failure_types[qn_id2]


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

                elif (res1_correct and (res2_invalid or res2_wrong)) or (res2_correct and (res1_invalid or res1_wrong)):
                    inconsistency_count += 1
                    inconsistency_dict = self.obtain_inconsistency_difference(row1, row2)
                    inconsistency_distance += inconsistency_dict.get('inconsistency_distance', 0)

                elif res1_wrong and res2_wrong:
                    inconsistency_dict = self.obtain_inconsistency_difference(row1, row2)
                    if inconsistency_dict['inconsistency_exists']:
                        inconsistency_count += 1
                    inconsistency_distance += inconsistency_dict.get('inconsistency_distance', 0)

                elif (res1_invalid and res2_wrong) or (res1_wrong and res2_invalid):
                    inconsistency_count += 1
                    # no need to check inconsistency distance

                else:
                    continue

                total_comparisons += 1
        
        return {
            "inconsistency_count": inconsistency_count,
            "total_comparisons": total_comparisons,
            "inconsistency_distance": inconsistency_distance,
            "total_correct" : total_correct,
            "total_assertion": total_assertion,
            "total_invalid": total_invalid
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
