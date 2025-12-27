
import os
import pandas as pd
from typing import Dict, Tuple
from utility.data_log_functions import DataLogHelper
from utility.constants import CodeGeneration, BigCodeBench

def compare_logs_against_no_mutation(res_dir: str):
    """
    This function compares all LLM output logs located in res_dir with the no_mutation LLM output log.

    Args:
        res_dir: directory to the csv files. this directory should also contain a "no_mutation" log output file
        task: the task type (e.g.: code generation, input prediction, etc)
        filter: strings that should be inside the log names of the csv log outputs
        anti-filter: strings that should NOT be inside the log names of the csv log outputs

    Returns:
        results_df: Pandas Dataframe containing inconsistency scores 
        category_dict: Python Dictionary containing aggegated scores
    """
    csv_logs = [f for f in os.listdir(res_dir) if (os.path.isfile(os.path.join(res_dir, f)) and 
                                                   f.endswith(".csv") and
                                                   "BigCodeBench" in f
                                                   )]

    csv_logs.sort()
    target_log_name = [l for l in csv_logs if "no_mutation" in l][-1]
    csv_logs.pop(csv_logs.index(target_log_name))
    target_log_path = os.path.join(res_dir, target_log_name)
    target_log = pd.read_csv(target_log_path)

    category_dict = {}
    mutation_dict = {}

    for log_name in csv_logs:

        log_category = DataLogHelper.obtain_category(log_name)
                
        log2_file_path = os.path.join(res_dir, log_name)
        log2 = pd.read_csv(log2_file_path) 

        target_log, log2 = DataLogHelper.standardize_two_df(target_log, log2)

        inconsistency_dict = DataLogHelper.compare_code_generation_dataframe_results(log1=target_log, log2=log2, task = CodeGeneration.NAME, benchmark = BigCodeBench.NAME)
        
        # Adding results into the dataframe
        cleaned_mutation_name = DataLogHelper.clean_up_csv_name(log_name.replace('.csv', ''))

        # Metrics for model inconsistency calculation
        mutation_inconsistencies = inconsistency_dict['total_inconsistencies']
        mutation_cumulative_inc_dist = inconsistency_dict['cumulative_inconsistency_distance']
        mutation_questions = inconsistency_dict['total_inconsistency_comparisons']

        # Metrics for model accuracy calculation
        mutation_successes = inconsistency_dict['log2_success']
        mutation_answered = inconsistency_dict['log2_total_answered']

        # Metrics for direction calculation
        mutation_incorrect_dir = inconsistency_dict.get('incorrect_dir', {}) or {}
        mutation_invalid_dir = inconsistency_dict.get('invalid_dir', {}) or {}

        if log_category:
            d: Dict = category_dict.get(log_category, {})

            # Obtaining metrics used for model inconsistency
            d['total_inconsistencies'] = d.get('total_inconsistencies', 0) + mutation_inconsistencies
            d['total_questions'] = d.get('total_questions', 0) + mutation_questions
            d['cumulative_inconsistency_distance'] = d.get('cumulative_inconsistency_distance', 0) + mutation_cumulative_inc_dist

            # Obtaining metrics used for calculating model accuracy
            d['total_success'] = d.get('total_success', 0) + mutation_successes
            d['total_answered'] = d.get('total_answered', 0) + mutation_answered

            # Accounting for incorrect and invalid directions (merge-by-sum)
            incorrect_dir: Dict = d.get('incorrect_dir', {}) or {}
            d['incorrect_dir'] = {
                k: incorrect_dir.get(k, 0) + mutation_incorrect_dir.get(k, 0)
                for k in (incorrect_dir | mutation_incorrect_dir)
            }

            invalid_dir: Dict = d.get('invalid_dir', {}) or {}
            d['invalid_dir'] = {
                k: invalid_dir.get(k, 0) + mutation_invalid_dir.get(k, 0)
                for k in (invalid_dir | mutation_invalid_dir)
            }

            category_dict[log_category] = d

        # adding results in mutation_dict, with the mutation name as key
        mutation_dict[cleaned_mutation_name] = {
            'total_inconsistencies': mutation_inconsistencies,
            'total_questions': mutation_questions,
            'total_success': mutation_successes,
            'total_answered': mutation_answered,
            'cumulative_inconsistency_distance': mutation_cumulative_inc_dist,
            'incorrect_dir': dict(mutation_incorrect_dir),
            'invalid_dir': dict(mutation_invalid_dir),
        }

    return mutation_dict | category_dict
        

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    results_dict = compare_logs_against_no_mutation(args.res_dir)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: save category_dict
    with open(args.out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
