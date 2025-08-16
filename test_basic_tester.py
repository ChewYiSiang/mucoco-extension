import os
import sys

curr_dir = os.getcwd()
sys.path.append(curr_dir)

from llm_models.code_llms import Mistral
from code_inconsistency.code_inconsistency_tester import LLMConsistencyTester
from code_inconsistency.prompt_templates.prompt_template import CodeInconsistencyPromptTemplate

def test_demorgan_mutation():
    """
    Test to verify that the DeMorgan mutation is correctly applied by the test framework.
    This tests the mutation functionality, not LLM responses.
    """
    print("Testing DeMorgan mutation application...")
    
    # Initialize tester 
    llmtester = LLMConsistencyTester("HumanEval_Input_Output")
    
    # Get a document that has boolean operations suitable for DeMorgan
    test_doc_id = "HumanEvalTF144"
    
    # Get the original code
    original_doc = llmtester.question_database.find_one({"_id": test_doc_id})
    if original_doc is None:
        print(f"Document {test_doc_id} not found in database")
        return False
    original_code = original_doc['full_sol']
    
    print(f"Original code:\n{original_code}")
    
    # Apply DeMorgan mutation
    from code_mutation.mutation_functions import CodeMutator
    
    try:
        mutated_code = CodeMutator.mutate_demorgan(original_code)
        
        if mutated_code == original_code:
            print("FAIL: No mutation was applied - code is identical")
            return False
        
        print(f"\nMutated code:\n{mutated_code}")
        
        # Test both original and mutated code with the test data
        input_args = original_doc['input']['args']
        output_args = original_doc['output']['args']
        func_name = "fizz_buzz"  # From the code we can see this is the function name
        
        print(f"\nTesting both versions with input: {input_args}, expected output: {output_args}")
        
        # Test original code
        try:
            CodeMutator.check_solution_validity(original_code, output_args, input_args, func_name)
            print("✓ Original code passes test")
            original_passes = True
        except Exception as e:
            print(f"✗ Original code fails test: {type(e).__name__}: {e}")
            original_passes = False
        
        # Test mutated code  
        try:
            CodeMutator.check_solution_validity(mutated_code, output_args, input_args, func_name)
            print("✓ Mutated code passes test")
            mutated_passes = True
        except Exception as e:
            print(f"✗ Mutated code fails test: {type(e).__name__}: {e}")
            mutated_passes = False
            
        # Check semantic equivalence
        semantic_equivalent = CodeMutator.check_semantic_equivalence(
            original_code, mutated_code, input_args, func_name
        )
        print(f"Semantic equivalence: {'✓' if semantic_equivalent else '✗'}")
        
        if original_passes == mutated_passes and semantic_equivalent:
            print("\nSUCCESS: DeMorgan mutation is working correctly!")
            print("Both versions have same test result and are semantically equivalent")
            return True
        elif not original_passes and not mutated_passes and semantic_equivalent:
            print("\nSUCCESS: Both codes fail but are semantically equivalent (test issue, not mutation issue)")
            return True
        else:
            print("\nFAIL: Mutation changed behavior unexpectedly")
            return False
        
    except Exception as e:
        print(f"ERROR during mutation: {e}")
        return False

if __name__ == "__main__":
    success = test_demorgan_mutation()
    print(f"Test {'PASSED' if success else 'FAILED'}")