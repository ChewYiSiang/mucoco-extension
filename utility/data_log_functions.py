
import pandas as pd
import os
from utility.constants import CodeGeneration, HumanEval, BigCodeBench
from typing import Tuple, Dict, Any, List
from code_generation.code_generation_tester import LLMExecutionRuntimeError, LLMExecutionError
from prediction_inconsistency.utility.database_helper import extract_assert_cases
from utility.custom_decorators import multiprocessing_method
from utility.benchmark_helper import HumanEvalHelper, BigCodeBenchHelper
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI)


LLM_EXECUTION_ERROR = "LLM Execution Error"
LLM_CORRECTNESS_ERROR = "LLM Correctness Error"
LLM_ANSWER_CORRECT = "LLM Answer Correct"
MUTATED_TASK = "Mutated"
ORIGINAL_TASK = "Original"

CORRECTNESS_INCONSISTENCY = "Correctness Inconsistency"
INCORRECTNESS_INCONSISTENCY = "Incorrectness Inconsistency"
INVALID_INCONSISTENCY = "Invalid Inconsistency"


class TimeoutError(Exception): pass

class DataLogHelper:
    def __init__(self):
        self.inconsistency_direction = {
            "original": 0,
            "mutated": 0,
        }

    def compare_model_outputs(
        log1_data: Dict[str, Any],
        log2_data: Dict[str, Any],
        task: str, 
        benchmark: str, 
    ) -> Dict[str, float | bool]:
        """
        This function is used to compare the model outputs from 2 LLM output logs

        Args:
            - log1_data: dictionary containing llm output data 
            - log2_data: dictionary containing llm output data 
            - task: task type (e.g.: code_generation etc)
            - benchmark: benchmark name (e.g.: humaneval, bigcodebench etc)

        Returns:
            - {
            'inconsistency_exists' (bool): True if an inconsistency exists between the two LLM outputs else False
            'inconsistency_score' (float): inconsistency distance metric between the 2 LLM outputs
            }
        """
        # Declaring variables
        model_output1 = log1_data['model_output']
        model_output2 = log2_data['model_output']

        task_id = log1_data['task_id']

        prompt_1 = log1_data['prompt']
        prompt_2 = log2_data['prompt']

        inconsistency_dict = {}
        
        # calculating inconsistency metrics
        if task == CodeGeneration.NAME:
            check_function = log1_data['check_function']

            if task_id == "HumanEvalo32":
                inconsistency_score1 = HumanEvalHelper.humaneval32_inconsistency_score(prompt_1, model_output1)
                inconsistency_score2 = HumanEvalHelper.humaneval32_inconsistency_score(prompt_2, model_output2)
                
                inconsistency_dict['inconsistency_exists'] = inconsistency_score1 == inconsistency_score2
                inconsistency_dict['inconsistency_distance'] = abs(inconsistency_score1-inconsistency_score2)/100
                return inconsistency_dict

            elif benchmark == HumanEval.NAME:
                _, test_cases, _ = extract_assert_cases(check_function)
                input_data = []
                output_data = []
                for inputs, output in test_cases:
                    output_data.append(output)
                    input_data.append(inputs)
                
                inconsistency_results1 = HumanEvalHelper.humaneval_inconsistency_score(
                    prompt = prompt_1, 
                    candidate = model_output1, 
                    test_inputs = input_data, 
                    test_outputs = output_data,
                )
                
                inconsistency_results2 = HumanEvalHelper.humaneval_inconsistency_score(
                    prompt = prompt_2, 
                    candidate = model_output2, 
                    test_inputs = input_data, 
                    test_outputs = output_data,
                )

            else:
                check_function = log1_data['check_function']

                inconsistency_results1 = BigCodeBenchHelper.bigcodebench_inconsistency_score(
                    candidate = model_output1, 
                    test_suite = check_function, 
                    task_id = task_id
                )
                    
                inconsistency_results2 = BigCodeBenchHelper.bigcodebench_inconsistency_score(
                    candidate = model_output2, 
                    test_suite = check_function, 
                    task_id = task_id
                )

            # if statement catching cases where the results returned are both empty dictionaries
            # check BigCodeBencho174 as an example
            if any(len(res) == 0 for res in (inconsistency_results1, inconsistency_results2)):
                inconsistency_dict['inconsistency_exists'] = False
                inconsistency_dict['inconsistency_distance'] = 0
                return inconsistency_dict

            inconsistency_distance = 0
            for key in inconsistency_results1.keys():
                res1 = inconsistency_results1[key]
                res2 = inconsistency_results2.get(key, False)
                inconsistency_distance += 1 if res1 != res2 else 0

            inconsistency_dict['inconsistency_exists'] = inconsistency_results1 != inconsistency_results2
            inconsistency_dict['inconsistency_distance'] = inconsistency_distance/ max(inconsistency_distance, len(inconsistency_results1))
        else:
            inconsistency_dict['inconsistency_exists'] = model_output1 != model_output2
        return inconsistency_dict
    
    def verify_failure_type_relevance(res: str | float) -> Dict[str, bool]:
        """
        this method determines if the LLM output failure string is a relevant output for inconsistency testing
        relevant cases includes:
            1. output is of type float -> LLM output is valid and correct
            2. Assertion Error -> LLM output is valid but incorrect
            3. Unable to parse the LLM answer -> LLM output is invalid
            4. LLMExecutionRuntimeError -> LLM failing to complete the task in the allotted time 
        
        irrelevant cases includes:
            1. Original task could not be mutated by the framework
            2. Original task did not have more than 1 example, hence disqualifying it as an appropriate task for few shot prompting.
        """

        failure_dict = {
            LLM_ANSWER_CORRECT: isinstance(res, float),
            LLM_CORRECTNESS_ERROR: False,
            LLM_EXECUTION_ERROR: False
        }

        if isinstance(res, str):
            has_assertion_error = AssertionError.__name__ in res and "Mutation" not in res
            failed_to_parse_llm_ans = "could_not_parse_LLM_answer" in res
            has_llm_runtime_error = (LLMExecutionRuntimeError.__name__ in res) or (LLMExecutionError.__name__ in res)

            failure_dict[LLM_CORRECTNESS_ERROR] = has_assertion_error or failed_to_parse_llm_ans
            failure_dict[LLM_EXECUTION_ERROR] = has_llm_runtime_error

        return failure_dict
    
    def inconsistency_heuristics(
            self, 
            task: str, 
            benchmark: str,
            log1_data: Dict[str, Any],
            log2_data: Dict[str, Any],
            ) -> Dict[str, Any]:
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
            - benchmark: name of the dataset (e.g.: humaneval, bigcodebench etc)
        Returns:
            - dictionary containing inconsistencies scores
        """

        inconsistencies = {
            "incorrect_dir": [None],
            "invalid_dir": [None],
            "total_inconsistencies": 0,
            "inconsistency_comparison": 0,
            'inconsistency_distance': 0
        }

        if log1_data['task_id'] in( "BigCodeBencho471",): 
            return inconsistencies

        # Extracting key variables from log data
        failure_type1 = log1_data['failure_type']
        failure_type2 = log2_data['failure_type']

        res1_result = DataLogHelper.verify_failure_type_relevance(failure_type1)
        res2_result = DataLogHelper.verify_failure_type_relevance(failure_type2)

        # determining the outcome of the llm output: either correct or incorrect
        res1_correct = res1_result[LLM_ANSWER_CORRECT]
        res2_correct = res2_result[LLM_ANSWER_CORRECT]
        res1_wrong = res1_result[LLM_CORRECTNESS_ERROR]
        res2_wrong = res2_result[LLM_CORRECTNESS_ERROR]
        res1_invalid = res1_result[LLM_EXECUTION_ERROR]
        res2_invalid = res2_result[LLM_EXECUTION_ERROR]

        def run_comparison():
            return DataLogHelper.compare_model_outputs(
                log1_data=log1_data,
                log2_data=log2_data,
                task=task,
                benchmark = benchmark
            )

        # 1. check if both llm outputs are correct > no inconsistencies
        if res1_correct and res2_correct:
            inconsistency_errors = {}

        else:

            # 2. check if either res1 is correct and res2 is invalid or wrong
            if res1_correct and (res2_wrong or res2_invalid):
                if res2_wrong: inconsistencies['incorrect_dir'] = [MUTATED_TASK]
                if res2_invalid: inconsistencies['invalid_dir'] = [MUTATED_TASK]

            # 3. check if either res1 is wrong while res2 is correct
            elif res2_correct and (res1_wrong or res1_invalid):
                if res1_wrong: inconsistencies['incorrect_dir'] = [ORIGINAL_TASK]
                if res1_invalid: inconsistencies['invalid_dir'] = [ORIGINAL_TASK]

            elif (res1_wrong and res2_wrong):
                inconsistencies['incorrect_dir'] = [ORIGINAL_TASK, MUTATED_TASK]

            elif res1_wrong and res2_invalid:
                inconsistencies['incorrect_dir'] = [ORIGINAL_TASK]
                inconsistencies['invalid_dir'] = [MUTATED_TASK]

            elif res1_invalid and res2_wrong:
                inconsistencies['incorrect_dir'] = [MUTATED_TASK]
                inconsistencies['invalid_dir'] = [ORIGINAL_TASK]

            elif res1_invalid and res2_invalid:
                inconsistencies['invalid_dir'] = [ORIGINAL_TASK, MUTATED_TASK] 

            # else, either of them have some other errors that do not contribute to inconsistency
            else:
                return inconsistencies
        
            inconsistency_errors = run_comparison()

        
        inconsistencies['inconsistency_comparison'] += 1
        if inconsistency_errors.get('inconsistency_exists', False):
            inconsistencies['total_inconsistencies'] = 1
            if res1_invalid or res2_invalid:
                inconsistencies['inconsistency_type'] = INVALID_INCONSISTENCY
            elif res1_wrong and res2_wrong:
                inconsistencies['inconsistency_type'] = INCORRECTNESS_INCONSISTENCY
            else:
                inconsistencies['inconsistency_type'] = CORRECTNESS_INCONSISTENCY


        inconsistencies['inconsistency_distance'] = inconsistency_errors.get('inconsistency_distance', 0)
        
        return inconsistencies

    @staticmethod
    def compare_code_generation_dataframe_results(
        log1: pd.DataFrame, 
        log2: pd.DataFrame, 
        task: str, 
        benchmark: str
    ) -> Dict[str, int | Dict]:
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
        
        total_comparisons = 0             
        log1_total_answered = 0
        log2_total_answered = 0
        total_inconsistencies = 0
        cumulative_inconsistency_distance = 0

        incorrect_dir_dict = {}
        invalid_dir_dict = {}

        total_tasks = log1.shape[0]

        inconsistency_types = {
            INCORRECTNESS_INCONSISTENCY: 0,
            CORRECTNESS_INCONSISTENCY: 0,
            INVALID_INCONSISTENCY: 0,
        }

        datalog_helper = DataLogHelper()

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
            inconsistency_scores = datalog_helper.inconsistency_heuristics(
                log1_data = log1_data,
                log2_data = log2_data,
                task = task,
                benchmark = benchmark
            )
            
            # Obtaining useful metrics for inconsistency scoring
            incorrect_dir = inconsistency_scores['incorrect_dir']
            for inc in incorrect_dir:
                if inc:
                    incorrect_dir_dict[inc] = incorrect_dir_dict.get(inc, 0) + 1

            invalid_dir = inconsistency_scores['invalid_dir']
            for inv in invalid_dir:
                if inv:
                    invalid_dir_dict[inv] = invalid_dir_dict.get(inv, 0) + 1


            total_inconsistencies += inconsistency_scores['total_inconsistencies']
            inconsistency_type = inconsistency_scores.get('inconsistency_type', None)
            if inconsistency_type:
                inconsistency_types[inconsistency_type] +=1 
            total_comparisons += inconsistency_scores['inconsistency_comparison']

            if task == CodeGeneration.NAME:
                cumulative_inconsistency_distance += inconsistency_scores['inconsistency_distance']

        mask1 = log1_orig['failure_type'].map(type).eq(float)
        mask2 = log2_orig['failure_type'].map(type).eq(float)

        return {
            'incorrect_dir' : incorrect_dir_dict,
            'invalid_dir' : invalid_dir_dict,
            'total_inconsistencies': total_inconsistencies,
            'inconsistency_types': inconsistency_types,
            'total_inconsistency_comparisons': total_comparisons,
            'cumulative_inconsistency_distance': cumulative_inconsistency_distance,
            'log1_success': int(mask1.sum()),
            'log2_success': int(mask2.sum()),
            'log1_total_answered': log1_total_answered,
            'log2_total_answered': log2_total_answered,
            'total_tasks': total_tasks
        }
    
    def standardize_two_df(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        common_ids = set(df1["task_id"]) & set(df2["task_id"])
        if not common_ids:
            print("⚠️ No matching task_ids found between the two DataFrames.")
            return df1.iloc[0:0], df2.iloc[0:0]  # return empty aligned frames

        df1_filtered = df1[df1["task_id"].isin(common_ids)].copy()
        df2_filtered = df2[df2["task_id"].isin(common_ids)].copy()

        df1_filtered = df1_filtered.drop_duplicates(subset=["task_id"], keep="first")
        df2_filtered = df2_filtered.drop_duplicates(subset=["task_id"], keep="first")

        df1_filtered = df1_filtered.sort_values("task_id").reset_index(drop=True)
        df2_filtered = df2_filtered.sort_values("task_id").reset_index(drop=True)

        return df1_filtered, df2_filtered
    
    def clean_up_csv_name(file_name: str)-> str:
        mutation_type = file_name.split("shot_")[-1]
        if "_" in mutation_type:
            mutation = mutation_type.replace("_", " ").title()
            return mutation
        return mutation_type.capitalize()
    
    def obtain_category(log_name:str) -> str | None:
        mutation_categories = {
            "Lexical": [
                "literal_format",
                "random",
                "sequential"
            ],
            "Syntactic": [
                "for2while",
                "for2enumerate"
            ],
            "Logical": [
                "boolean_literal",
                "constant_unfold_add",
                "constant_unfold_mult",
                "constant_unfold",
                "demorgan",
                "commutative_reorder"
            ]
        }

        for cat, mut in mutation_categories.items():
            for m in mut:
                if m in log_name:
                    return cat

        return None

