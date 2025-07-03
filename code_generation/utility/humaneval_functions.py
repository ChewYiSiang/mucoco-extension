
import ast
import doctest
from typing import Tuple, List, Dict
import builtins


class CodeGenerationHumanEvalHelper():

    def seperate_original_desciptions(prompt: str) -> Tuple[str, str] | None:
        tree = ast.parse(prompt)

        func_node = None
        for idx, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
                func_node = tree.body[idx]

        tree.body.remove(func_node)
        
        try:
            doc_string = ast.get_docstring(func_node) or None
            
            if (len(func_node.body) > 0 and                             # checks if there are any code in the func_node
                isinstance(func_node.body[0], ast.Expr) and             # checks that the first item in the func_node.body is indeed an expr, which would correspond with the docstring
                isinstance(func_node.body[0].value, ast.Constant) and   # checks if the value of the first item in the list is a constant
                isinstance(func_node.body[0].value.value, str)):        # checks if the type is a string

                func_node.body.pop(0)                                   # removing the doc string from the function

            tree.body.insert(idx, func_node)

            new_prog = ast.unparse(tree)

            return new_prog, doc_string

        except Exception as e:
            if type(e) == NameError:
                print("Failed to extraction proper function from the string.")
            else:
                print("Failed due to following error: {e}".format(e = e))
            return [None, None]
    
    def extract_examples(desc: str) -> Tuple[str, Dict[str, str]]:
        parser = doctest.DocTestParser()
        tests = parser.get_doctest(desc, {}, "test_cases", "tests", 0) # extracting all doc test cases

        test_cases = {}
        for test in tests.examples:
            test_cases[test.source.strip()] = test.want.strip()

        lines = desc.splitlines()

        if len(tests.examples) < 1:
            return (desc.strip(), "")

        remove_next_line = False        # since doctests consist of a function followed by the expected answer in the following line, this boolean stores if the next line should be removed or not
        idx = 0
        while idx < len(lines):
            curr_line = lines[idx]
            if curr_line.__contains__(">>>"):
                lines.pop(idx)
                remove_next_line = True
            elif remove_next_line:
                lines.pop(idx)
                remove_next_line = False
            else:
                idx+=1
        
        
        return ("\n".join(lines), test_cases)

    def process_original_tests(test_cases: str) -> str | None:
        """
        This function is used to process the original check function and remove any useless information.

        E.g.: HumanEval/0 original check function included an unnecessary METADATA dictionary.
            
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
        """
        tree = ast.parse(test_cases)
        test_case = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Dict):       # if statement checking if the value assignment is a dictionary and that the dictionary is not empty
                    end_no = node.end_lineno            # find the line number in which the dictionary ends at
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
        
    def check_test_case(test_case: str, code_snippet: str, func_name: str) -> bool:
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
        """
        t = ast.parse(code)                 # parsing the string code to obtain the AST
        for node in t.body:                 # for loop iterating through each node in the the AST
            if isinstance(node, ast.Expr):                      # if statement checking if the node is of type ast.Expr
                node_val = node.value
                if isinstance(node_val, ast.Call):              # if the node is calling a function
                    func_name = node_val.func.id                # obtaining the function name
                    func_args = node_val.args                   # obtaining the function args
                    if hasattr(builtins, func_name):          # if statement checking if the function is a built in python function. If so, this means that this function cannot be the "task function"
                        for arg in func_args:                   
                            if isinstance(arg, ast.Call):       
                                subnode = ast.unparse(arg)
                                return CodeGenerationHumanEvalHelper.extract_func_name_from_example(subnode)
                    else:
                        return func_name
            else:
                raise ValueError("Could not extract the function name.")