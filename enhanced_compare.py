import pandas as pd 
import sys
import os
import glob
from collections import defaultdict
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utility.data_log_functions import DataLogHelper

def categorize_failure(failure_value):
    """Categorize failure types into meaningful groups"""
    if pd.isna(failure_value) or isinstance(failure_value, float):
        return "SUCCESS"
    
    failure_str = str(failure_value).strip()
    
    # Check for empty/blank strings (successful mutations)
    if failure_str == '' or failure_str.lower() == 'nan':
        return "SUCCESS"
    
    # Check for MutationFailedError
    if "MutationFailedError" in failure_str:
        return "UNMUTABLE_MutationFailed"
    
    # Mutation unavailable errors (not actual mutation errors)
    if "NoBooleanLiteralError" in failure_str:
        return "UNMUTABLE_NoBooleanLiteral"
    elif "IdenticalMutationError" in failure_str:
        return "UNMUTABLE_Identical"
    elif "NoCommutativeOperationError" in failure_str:
        return "UNMUTABLE_NoCommutative"
    elif "NoConstantUnfoldError" in failure_str:
        return "UNMUTABLE_NoConstant"
    elif "NoBooleanOperationError" in failure_str:
        return "UNMUTABLE_NoDeMorgan"
    elif "NoForLoopError" in failure_str:
        return "UNMUTABLE_NoForLoop"
    elif "NoLiteralFormatError" in failure_str:
        return "UNMUTABLE_NoLiteralFormat"
    
    # Runtime errors
    elif "SyntaxError" in failure_str:
        return "RUNTIME_ERROR_Syntax"
    elif "TimeoutError" in failure_str:
        return "RUNTIME_ERROR_Timeout"
    elif "IndentationError" in failure_str:
        return "RUNTIME_ERROR_Indentation"
    elif "NameError" in failure_str:
        return "RUNTIME_ERROR_Name"
    elif "TypeError" in failure_str:
        return "RUNTIME_ERROR_Type"
    elif "ValueError" in failure_str:
        return "RUNTIME_ERROR_Value"
    
    # Logic errors
    elif "AssertionError" in failure_str:
        return "LOGIC_ERROR_Assertion"
    
    # Other
    else:
        return f"OTHER_ERROR_{failure_str[:50]}"

def enhanced_comparison(reference_df, comparison_df, reference_name, comparison_name):
    """Enhanced comparison with comprehensive categorization"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED COMPARISON: {reference_name} vs {comparison_name}")
    
    # Stats tracking
    stats = {
        'total_tasks': 0,
        'mutation_error_tasks': 0,
        'mutable_tasks': 0,
        'both_succeeded': 0,
        'both_failed_assertion': 0,
        'both_failed_other': 0,
        'ref_succeeded_comp_failed': 0,
        'ref_failed_comp_succeeded': 0,
        'error_categories': defaultdict(int),
        'task_details': []
    }
    
    # Process each task in reference
    for _, ref_row in reference_df.iterrows():
        task_id = ref_row['task_id']
        
        # Find matching task in comparison
        comp_matches = comparison_df[comparison_df['task_id'] == task_id]
        if len(comp_matches) != 1:
            continue
            
        comp_row = comp_matches.iloc[0]
        stats['total_tasks'] += 1
        
        # Categorize results
        ref_category = categorize_failure(ref_row['failure_type'])
        comp_category = categorize_failure(comp_row['failure_type'])
        
        # Track error categories for reporting
        if ref_category != "SUCCESS":
            stats['error_categories'][ref_category] += 1
        if comp_category != "SUCCESS":
            stats['error_categories'][comp_category] += 1
        
        # Check if task has unmutable errors (exclude from main analysis)
        has_unmutable_error = (ref_category.startswith("UNMUTABLE") or 
                              comp_category.startswith("UNMUTABLE"))
        
        if has_unmutable_error:
            stats['mutation_error_tasks'] += 1
            outcome = "UNMUTABLE_EXCLUDED"
        else:
            # This is a mutable task - include in main analysis
            stats['mutable_tasks'] += 1
            
            # Categorize outcomes for mutable tasks only
            if ref_category == "SUCCESS" and comp_category == "SUCCESS":
                stats['both_succeeded'] += 1
                outcome = "BOTH_SUCCEEDED"
            elif ref_category != "SUCCESS" and comp_category != "SUCCESS":
                # Both failed - categorize by error type
                if ref_category.startswith("LOGIC_ERROR") and comp_category.startswith("LOGIC_ERROR"):
                    stats['both_failed_assertion'] += 1
                    outcome = "BOTH_FAILED_ASSERTION"
                else:
                    stats['both_failed_other'] += 1
                    outcome = "BOTH_FAILED_OTHER"
            elif ref_category == "SUCCESS":
                stats['ref_succeeded_comp_failed'] += 1
                outcome = "INCONSISTENCY_REF_SUCCESS"
            else:
                stats['ref_failed_comp_succeeded'] += 1
                outcome = "INCONSISTENCY_COMP_SUCCESS"
            
        # Store task details
        stats['task_details'].append({
            'task_id': task_id,
            'outcome': outcome,
            'ref_category': ref_category,
            'comp_category': comp_category,
            'ref_failure': str(ref_row['failure_type']) if pd.notna(ref_row['failure_type']) else None,
            'comp_failure': str(comp_row['failure_type']) if pd.notna(comp_row['failure_type']) else None
        })
    
    return stats

def save_results_to_csv(all_results, reference_name, output_file="comparison_results.csv"):
    """Save comparison results to CSV file"""
    
    # Prepare data for CSV
    csv_data = []
    
    for comparison_name, stats in all_results.items():
        total = stats['total_tasks']
        mutable = stats['mutable_tasks']
        mutation_errors = stats['mutation_error_tasks']
        
        # Calculate percentages
        mutable_pct = (mutable/total*100) if total > 0 else 0
        mutation_error_pct = (mutation_errors/total*100) if total > 0 else 0
        
        # Calculate inconsistency metrics
        total_inconsistencies = stats['ref_succeeded_comp_failed'] + stats['ref_failed_comp_succeeded']
        both_succeeded_count = stats['both_succeeded']
        tasks_with_at_least_one_success = both_succeeded_count + total_inconsistencies
        inconsistency_rate = (total_inconsistencies/tasks_with_at_least_one_success*100) if tasks_with_at_least_one_success > 0 else 0
        
        # Calculate individual percentages for mutable tasks
        both_succeeded_pct = (stats['both_succeeded']/mutable*100) if mutable > 0 else 0
        both_failed_assertion_pct = (stats['both_failed_assertion']/mutable*100) if mutable > 0 else 0
        both_failed_other_pct = (stats['both_failed_other']/mutable*100) if mutable > 0 else 0
        ref_success_comp_fail_pct = (stats['ref_succeeded_comp_failed']/mutable*100) if mutable > 0 else 0
        ref_fail_comp_success_pct = (stats['ref_failed_comp_succeeded']/mutable*100) if mutable > 0 else 0
        
        row = {
            'reference_file': reference_name,
            'comparison_file': comparison_name,
            'total_tasks': total,
            'mutation_error_tasks': mutation_errors,
            'mutation_error_pct': round(mutation_error_pct, 2),
            'mutable_tasks': mutable,
            'mutable_tasks_pct': round(mutable_pct, 2),
            'both_succeeded': stats['both_succeeded'],
            'both_succeeded_pct': round(both_succeeded_pct, 2),
            'both_failed_assertion': stats['both_failed_assertion'],
            'both_failed_assertion_pct': round(both_failed_assertion_pct, 2),
            'both_failed_other': stats['both_failed_other'],
            'both_failed_other_pct': round(both_failed_other_pct, 2),
            'ref_success_comp_fail': stats['ref_succeeded_comp_failed'],
            'ref_success_comp_fail_pct': round(ref_success_comp_fail_pct, 2),
            'ref_fail_comp_success': stats['ref_failed_comp_succeeded'],
            'ref_fail_comp_success_pct': round(ref_fail_comp_success_pct, 2),
            'total_inconsistencies': total_inconsistencies,
            'tasks_with_at_least_one_success': tasks_with_at_least_one_success,
            'inconsistency_rate_pct': round(inconsistency_rate, 2)
        }
        
        csv_data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    return output_file

def save_inconsistency_summary(all_results, reference_name, output_file="inconsistency_summary.csv"):
    """Save simplified inconsistency summary to CSV file"""
    
    # Prepare data for CSV
    csv_data = []
    
    for comparison_name, stats in all_results.items():
        total_inconsistencies = stats['ref_succeeded_comp_failed'] + stats['ref_failed_comp_succeeded']
        both_succeeded_count = stats['both_succeeded']
        tasks_with_at_least_one_success = both_succeeded_count + total_inconsistencies
        inconsistency_rate = (total_inconsistencies/tasks_with_at_least_one_success*100) if tasks_with_at_least_one_success > 0 else 0
        
        row = {
            'reference_file': reference_name,
            'comparison_file': comparison_name,
            'total_inconsistencies': total_inconsistencies,
            'tasks_with_at_least_one_success': tasks_with_at_least_one_success,
            'inconsistency_rate_pct': round(inconsistency_rate, 2)
        }
        
        csv_data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"Inconsistency summary saved to: {output_file}")
    return output_file

def print_detailed_stats(stats, reference_name, comparison_name):
    """Print comprehensive statistics"""
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    total = stats['total_tasks']
    mutable = stats['mutable_tasks']
    mutation_errors = stats['mutation_error_tasks']
    
    print(f"Total Tasks: {total}")
    print(f"Tasks with Unmutable Errors: {mutation_errors} ({mutation_errors/total*100:.2f}%)")
    print(f"Mutable Tasks (analyzed): {mutable} ({mutable/total*100:.2f}%)")
    
    if mutable > 0:
        print(f"\n{'='*40}")
        print(f"ANALYSIS OF {mutable} MUTABLE TASKS:")
        print(f"{'='*40}")
        print(f"Both Succeeded: {stats['both_succeeded']} ({stats['both_succeeded']/mutable*100:.2f}%)")
        print(f"Both Failed (Assertion Errors): {stats['both_failed_assertion']} ({stats['both_failed_assertion']/mutable*100:.2f}%)")
        print(f"Both Failed (Other Errors): {stats['both_failed_other']} ({stats['both_failed_other']/mutable*100:.2f}%)")
        print(f"⚠️  {reference_name} Success, {comparison_name} Failed: {stats['ref_succeeded_comp_failed']} ({stats['ref_succeeded_comp_failed']/mutable*100:.2f}%)")
        print(f"⚠️  {comparison_name} Success, {reference_name} Failed: {stats['ref_failed_comp_succeeded']} ({stats['ref_failed_comp_succeeded']/mutable*100:.2f}%)")
        
        # Debug: Check if counts add up
        total_outcomes = (stats['both_succeeded'] + stats['both_failed_assertion'] + 
                         stats['both_failed_other'] + stats['ref_succeeded_comp_failed'] + 
                         stats['ref_failed_comp_succeeded'])
        print(f"DEBUG: Total outcome counts: {total_outcomes} (should equal {mutable})")
        if total_outcomes != mutable:
            print(f"ERROR: Counts don't match! Difference: {total_outcomes - mutable}")
        
        # Inconsistencies (traditional metric) - clearer explanation
        total_inconsistencies = stats['ref_succeeded_comp_failed'] + stats['ref_failed_comp_succeeded']
        both_succeeded_count = stats['both_succeeded']
        tasks_with_at_least_one_success = both_succeeded_count + total_inconsistencies
        
        print(f"\nTraditional Inconsistency Metric (tasks where at least one succeeded):")
        print(f"   Both succeeded: {both_succeeded_count}")
        print(f"   + Inconsistencies (one succeeded, one failed): {total_inconsistencies}")
        print(f"   = Tasks with at least one success: {tasks_with_at_least_one_success}")
        if tasks_with_at_least_one_success > 0:
            print(f"   Inconsistency Rate: {total_inconsistencies}/{tasks_with_at_least_one_success} ({total_inconsistencies/tasks_with_at_least_one_success*100:.2f}%)")
        else:
            print(f"   Inconsistency Rate: {total_inconsistencies}/{tasks_with_at_least_one_success} (No tasks with success)")
    
    print(f"\n{'='*40}")
    print("ERROR CATEGORY ANALYSIS:")
    print(f"{'='*40}")
    
    # Group by error type
    unmutable_errors = {k: v for k, v in stats['error_categories'].items() if k.startswith('UNMUTABLE')}
    runtime_errors = {k: v for k, v in stats['error_categories'].items() if k.startswith('RUNTIME_ERROR')}
    logic_errors = {k: v for k, v in stats['error_categories'].items() if k.startswith('LOGIC_ERROR')}
    other_errors = {k: v for k, v in stats['error_categories'].items() if k.startswith('OTHER_ERROR')}
    
    if unmutable_errors:
        print(f"\nUnmutable Errors (no mutation possible):")
        for error_type, count in sorted(unmutable_errors.items()):
            print(f"   {error_type}: {count}")
    
    if runtime_errors:
        print(f"\nRuntime Errors:")
        for error_type, count in sorted(runtime_errors.items()):
            print(f"   {error_type}: {count}")
    
    if logic_errors:
        print(f"\nLogic Errors:")
        for error_type, count in sorted(logic_errors.items()):
            print(f"   {error_type}: {count}")
    
    if other_errors:
        print(f"\nOther Errors:")
        for error_type, count in sorted(other_errors.items()):
            print(f"   {error_type}: {count}")
    
    print(f"\n{'='*40}")
    print("IDENTICAL FAILURE ANALYSIS:")
    print(f"{'='*40}")
    # Note: This section had an error in the original code - stats doesn't have 'identical_error_types'
    print("   Analysis not available (needs to be implemented)")

def main():
    # Reference file (the one to compare all others against)
    reference_path = "results/Gemma/CruxEval_output_gemma/CruxEval_zero_shot_no_mutation.csv"
    
    # Directory containing files to compare
    comparison_dir = "results/Gemma/CruxEval_output_gemma"
    
    print("=" * 80)
    print("ENHANCED LLAMA INPUT PREDICTION COMPARISON")
    print("=" * 80)
    print(f"Reference file: {reference_path}")
    print(f"Comparing all CSV files in: {comparison_dir}")
    
    # Load reference file
    try:
        reference_df = pd.read_csv(reference_path)
        print(f"\nReference file shape: {reference_df.shape}")
    except FileNotFoundError:
        print(f"Error: Reference file not found at {reference_path}")
        return
    
    # Find all CSV files
    csv_files = []
    for root, _, files in os.walk(comparison_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    # Remove reference file from comparison list
    csv_files = [f for f in csv_files if f != reference_path]
    
    print(f"\nFound {len(csv_files)} CSV files to compare:")
    
    # Store all results for summary
    all_results = {}
    
    # Compare each file with the reference
    for file_path in sorted(csv_files):
        try:
            comparison_df = pd.read_csv(file_path)
            
            reference_name = os.path.basename(reference_path).replace('.csv', '')
            comparison_name = os.path.basename(file_path).replace('.csv', '')
            
            stats = enhanced_comparison(reference_df, comparison_df, reference_name, comparison_name)
            print_detailed_stats(stats, reference_name, comparison_name)
            
            all_results[comparison_name] = stats
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Final summary across all comparisons
    print(f"\n{'='*80}")
    print("CROSS-COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for name, stats in all_results.items():
        total = stats['total_tasks']
        mutable = stats['mutable_tasks']
        inconsistencies = stats['ref_succeeded_comp_failed'] + stats['ref_failed_comp_succeeded']
        comparable = stats['both_succeeded'] + inconsistencies
        
        print(f"{name}:")
        print(f"  Mutable: {mutable}/{total} ({mutable/total*100:.2f}%)")
        print(f"  Inconsistencies: {inconsistencies}/{comparable} ({inconsistencies/comparable*100:.2f}% if comparable > 0)")
        print()
    
    # Save results to CSV
    if all_results:
        reference_name = os.path.basename(reference_path).replace('.csv', '')
        output_csv = f"enhanced_comparison_{reference_name}.csv"
        save_results_to_csv(all_results, reference_name, output_csv)
        
        # Save simplified inconsistency summary
        output_summary_csv = f"inconsistency_summary_{reference_name}.csv"
        save_inconsistency_summary(all_results, reference_name, output_summary_csv)

if __name__ == "__main__":
    main()