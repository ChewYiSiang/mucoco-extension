
import ast
import doctest
from typing import Tuple, Dict
import builtins
from code_generation.utility.database_helper import DatabaseHelper
import multiprocessing


class CodeGenerationHumanEvalHelper(DatabaseHelper):
    """
    This class houses methods used as utility functions pertaining to the HumanEval dataset. The methods here are all static methods.

    Attributes:
        None
    """
    
    @staticmethod
    def extract_examples(desc: str) -> Tuple[str, Dict[str, str]]:
        """
        This function takes in the docstring extracted from a HumanEval prompt.

        For example, the original docstring extracted from HumanEval/0 is as follows:

            \""" Check if in given list of numbers, are any two numbers closer to each other than
            given threshold.
            >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
            False
            >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
            True
            \"""
        
        This function will then seperate the examples from the task description and store the examples into a dictionary. 
        This allows for easier reuse for other prompt techniques such as one shot/few shot prompting.

        Do note that the examples MUST be in standard doc test format in order for the doctest library to work as intended.

        Args:
            prompt (str): docstring from a HumanEval task
        
        Returns:
            tuple[str, dict[str, str]]:
                - str: the rest of the doc string with examples extracted out
                - dict[str, str]: dictionary with the function call and its input as keys and the expected output as the dictionary value
        """
        parser = doctest.DocTestParser()                               
        tests = parser.get_doctest(desc, {}, "test_cases", "tests", 0)  # extracting all doc test cases

        if len(tests.examples) < 1:                                     # if there are no examples, return the doc string stripped of white spaces and an empty dictionary
            return (desc.strip(), {})

        test_cases = {}                                                 # dictionary to store the examples and expected output
        for test in tests.examples:                                     # for loop iterating through each test examples and storing them in the test case dictionary
            test_cases[test.source.strip()] = test.want.strip()

        lines = desc.splitlines()                                       # splitting the docstring into seperate lines in a list

        remove_next_line = False                                        # since doctests consist of a function followed by the expected answer in the following line, this boolean stores if the next line should be removed or not
        idx = 0                                                         # pointer for the line index traversing through lines 
        while idx < len(lines):                                         # while loop iterating through entire list
            curr_line = lines[idx]
            if curr_line.__contains__(">>>"):                           # this indicates the start of a doc test
                lines.pop(idx)                                          
                remove_next_line = True                                 # remove_next_line set to true, indicating the removal of the following line, as per standard doc test format
            elif remove_next_line:                                      # checks if next line should be removed and removing it if so
                lines.pop(idx)
                remove_next_line = False
            else:                                                       # else, continue traversing through the list
                idx+=1
        
        return ("\n".join(lines), test_cases)                           # tuple containing the task description and dictionary containing the test cases 

    def process_original_tests(test_cases: str) -> str | None:
        """
        This function is used to process the original check function and remove any useless information. The check function can be directly extracted from the HumanEval dataset with no need for any pre-processing.

        E.g.: HumanEval/0 original check function included an unnecessary METADATA dictionary as seen below.
            
            METADATA = {
                'author': 'jt',
                'dataset': 'test'
            }

            def check(candidate):
                assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
                assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
                assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
                assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
                assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
                assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
                assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
            
        With this function, the METADATA dictionary is purged and only the check function is returned

        Args:
            test_cases (str): original check function in string format

        Returns:
            str: processed check function with no unnecessary information such as METADATA dictionaries
            None: returned if any errors were raised

        Raises:
            UnboundLocalError: Error is raised when the METADATA dictionary could not be properly extracted
            Exception: Error raised when it's any other errors
        """
        tree = ast.parse(test_cases)                                            # parsing the original check function to obtain the AST
        test_case = None                                                        # stores the check function during processing
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Dict):                            # if statement checking if the value assignment is a dictionary and that the dictionary is not empty
                    end_no = node.end_lineno                                    # find the line number in which the dictionary ends at
                    split_lines = test_cases.splitlines()
                    test_case =  "\n".join(split_lines[end_no+1:])
        original_test_case = ast.unparse(tree)
        test_case = original_test_case if test_case is None else test_case
        try:
            exec(test_case)
            return test_case
        except Exception as e:
            if isinstance(e, UnboundLocalError):
                print('Could not properly extract Dict from the code test case. Review this test case in the original database to troubleshoot.')
            else:
                print("Original test case could not be processed due to the following error: {e}".format(e = e))
            return None
        
    def check_test_case(
            test_case: str, 
            code_snippet: str, 
            func_name: str
        ) -> bool:
        """
        This function tests an input string code snippet against a given check function.

        Note that the test_case should be a check function.
        """
        namespace = {}
        try:
            exec(test_case, namespace)
            exec(code_snippet, namespace)
            namespace['check'](namespace[f'{func_name}'])
            return True
        except Exception as e:
            print("Canonical solution failed the test function due to following error: {e}".format(e = e))
            return False
    
    def process_llm_function_outputs(func_name : str, source_code: str) ->  Tuple[bool, str]:
        """
        This function is used to process the outputs from LLMs to remove unncessary print statements. It also returns a boolean that indicates if the syntax of the function name has been preserved. 
        """
        tree = ast.parse(source_code)
        lines = source_code.strip().splitlines()
        func_name_preserved = False
        lines_to_remove = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func_name_preserved = True
            elif isinstance(node, ast.ImportFrom):
                pass
            else:
                for i in range(node.lineno - 1, node.end_lineno):
                    lines_to_remove.add(i)
        
        processed_lines = [line for idx, line in enumerate(lines) if idx not in lines_to_remove]

        return (func_name_preserved, "\n".join(processed_lines))
    
    def extract_func_name_from_example(code: str) -> str | None:
        """
        Run this function on the test examples to obtain the true function names.

        This function is needed to circumvent the issue where there are more than 1 function in the given task and 
        we need to discern between the true task function and the helper function for testing

        E.g.: 
        test_case = 'round(find_zero([1, 2]), 2) # f(x) = 1 + 2x'
        extract_func_name(test_case) == 'find_zero'
        
        Args:
            code (str): The example extracted from question description

        Returns:
            str : The extracted function name

        """
        t = ast.parse(code)                 # parsing the string code to obtain the AST
        for node in t.body:                 # for loop iterating through each node in the the AST
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):                      
                func_name = node.value.func.id                # obtaining the function name
                func_args = node.value.args                   # obtaining the function args
                if hasattr(builtins, func_name):              # if statement checking if the function is a built in python function. If so, this means that this function cannot be the "task function"
                    for arg in func_args:                   
                        if isinstance(arg, ast.Call):       
                            subnode = ast.unparse(arg)
                            return CodeGenerationHumanEvalHelper.extract_func_name_from_example(subnode)
                else:
                    return func_name
            else:
                raise ValueError("Could not extract the function name.")
            
    @staticmethod
    def run_llm_answer(
        processed_output: str, 
        test_function: str, 
        func_name: str, 
        mp_queue: multiprocessing.Queue,
        ):
        namespace = {}

        try:
            exec(processed_output, namespace)
            exec(test_function, namespace)
            namespace['check'](namespace[func_name])
        except Exception as e:
            mp_queue.put(e)