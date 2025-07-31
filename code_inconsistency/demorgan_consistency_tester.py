from typing import List
from tqdm import tqdm
import copy
import pandas as pd
import os
import ast
from code_mutation.mutation_functions import CodeMutator
from code_inconsistency.prompt_templates.prompt_template import CodeInconsistencyPromptTemplate
from code_inconsistency.utility.humaneval_functions import CodeInconsistencyHumanEvalHelper


class DeMorganConsistencyTester:
    """
    Specialized tester for DeMorgan's law mutations that only tests specific documents 
    provided as input (already filtered for boolean operations).
    """

    @staticmethod 
    def run_demorgan_consistency_test(llmtester, llm, suitable_doc_ids: List[str], output_file_path: str) -> int:
        """
        Custom test runner that ONLY tests the provided document IDs (already filtered for DeMorgan suitability).
        
        Args:
            llmtester: LLMConsistencyTester instance
            llm: The language model to test
            suitable_doc_ids: List of document IDs that contain boolean operations
            output_file_path: Path to save CSV results
            
        Returns:
            int: Number of tests that passed
        """
        print(f"Running DeMorgan consistency test on {len(suitable_doc_ids)} specific documents...")
        
        task_pass_count = 0
        failed_mutations = []
        test_results = []
        
        for doc_id in tqdm(suitable_doc_ids):
            # Get the specific document by its ID
            qn_sample = llmtester.question_database.find_one({"_id": doc_id})
            if qn_sample is None:
                print(f"Document {doc_id} not found in database")
                continue
                
            # Extract data from document
            full_sol = qn_sample['full_sol']
            qn_desc = qn_sample['qn_desc'] 
            examples = qn_sample['examples']
            
            test_inputs = qn_sample['input']
            input_args = test_inputs['args']
            input_metadata = test_inputs['metadata']
            
            test_outputs = qn_sample['output']
            output_args = test_outputs['args']
            output_metadata = test_outputs['metadata']
            
            # Sanity check: Ensure the original solution passes the check function
            check_soln_validity = CodeInconsistencyHumanEvalHelper.check_database_answer(
                full_sol=full_sol,
                input_args=copy.deepcopy(input_args),
                input_metadata=input_metadata,
                output_args=output_args,
                output_metadata=output_metadata,
                examples=examples
            )
            
            # Processing of output args and metadata  
            output_args = ast.literal_eval(output_args) if output_metadata != str.__name__ else output_args
            
            if check_soln_validity is not True:
                print(f"⚠️  Skipping {doc_id} - original solution fails sanity check")
                
                # Log sanity check failure
                log_entry = {
                    "task_id": doc_id,
                    "prompt": None,
                    "model_output": None, 
                    "expected_output": {"args": output_args},
                    "failure_type": "invalid_full_solution"
                }
                DeMorganConsistencyTester.log_into_csv(output_file_path=output_file_path, input_data=log_entry)
                continue
            
            # Try to apply DeMorgan mutation
            try:
                mutated_dict = CodeMutator.mutate_for_code_inconsistency_test(
                    mutation_type="demorgan",
                    full_sol=full_sol,
                    examples=examples,
                    qn_desc=qn_desc,
                    input_args=copy.deepcopy(input_args),
                    output_args=output_args
                )
                
                mutated_sol = mutated_dict['full_sol']
                print(f"MUTATED CODE:")
                print(mutated_sol)
                print(f"=" * 50)
                print(f"✓ Successfully mutated {doc_id}")
                
                # Test with LLM
                prompt_template = CodeInconsistencyPromptTemplate.zero_shot_prompt()
                input_variables = {
                    'qn_desc': mutated_dict['qn_desc'],
                    'full_sol': mutated_sol,
                    'test_input': f'"{input_args}"' if isinstance(input_args, str) else input_args,
                    'example': None
                }
                
                ans = llm.invoke(input_variables=input_variables, prompt_template=prompt_template)
                
                # Process LLM answer (same as original tester)
                ans = DeMorganConsistencyTester.process_llm_ans(ans)
                
                # Check if answer matches expected output
                test_passed = False
                try:
                    assert ans == output_args
                    task_pass_count += 1
                    test_passed = True
                    print(f"✓ PASS for {doc_id}")
                except AssertionError:
                    print(f"✗ FAIL for {doc_id}: Expected {output_args}, got {ans}")
                
                # Log result to CSV (similar to original tester)
                log_entry = {
                    "task_id": doc_id,
                    "prompt": input_variables,
                    "model_output": (ans, type(ans)),
                    "expected_output": {"args": output_args},
                    "failure_type": None if test_passed else "AssertionError"
                }
                DeMorganConsistencyTester.log_into_csv(output_file_path=output_file_path, input_data=log_entry)
                    
            except Exception as e:
                print(f"✗ Failed to mutate {doc_id}: {type(e).__name__}: {e}")
                failed_mutations.append(doc_id)
                
                # Log mutation failure to CSV
                log_entry = {
                    "task_id": doc_id,
                    "prompt": None,
                    "model_output": None,
                    "expected_output": {"args": output_args},
                    "failure_type": f"MutationError: {type(e).__name__}: {e}"
                }
                DeMorganConsistencyTester.log_into_csv(output_file_path=output_file_path, input_data=log_entry)
                continue
        
        # Print summary
        print(f"\n" + "="*50)
        print(f"DeMorgan Consistency Test Results:")
        print(f"Total suitable documents: {len(suitable_doc_ids)}")
        print(f"Successfully mutated: {len(test_results)}")
        print(f"Failed to mutate: {len(failed_mutations)}")
        print(f"Tests passed: {task_pass_count}/{len(test_results)}")
        print(f"Success rate: {task_pass_count/len(test_results)*100:.1f}%" if len(test_results) > 0 else "N/A")
        print(f"="*50)
                
        return task_pass_count
    
    @staticmethod
    def process_llm_ans(ans):
        """
        Process LLM answer to match the format expected by the test.
        (Simplified version of the original method)
        """
        if isinstance(ans, str):
            ans = ans.strip()
            # Try to evaluate if it looks like a Python literal
            try:
                import ast
                return ast.literal_eval(ans)
            except (ValueError, SyntaxError):
                return ans
        return ans
    
    @staticmethod
    def log_into_csv(output_file_path: str, input_data: dict):
        """
        Log test result into CSV file (similar to original tester).
        
        Args:
            output_file_path: Path to the CSV file
            input_data: Dictionary containing test result data
        """
        df = pd.DataFrame([input_data])
        
        # Create file if it doesn't exist or append if it does
        if os.path.exists(output_file_path):
            df.to_csv(output_file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file_path, mode='w', header=True, index=False)