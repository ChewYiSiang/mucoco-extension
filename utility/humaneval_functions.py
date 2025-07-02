import ast
import doctest
from typing import Tuple, List, Dict
import inspect


class HumanEvalHelper():
    def __init__(self):
        pass

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
    

class CodeInconsistencyHumanEvalHelper(HumanEvalHelper):
    @staticmethod
    def check_input_output(
        full_sol: str, 
        test_input: str, 
        expected_output: str, 
        func_name: str, 
        input_metadata: List[str]
    ) -> bool:
        namespace = {}
        try:
            exec(full_sol, namespace)
            # sig = inspect.signature(namespace[func_name])
            # if len(sig.parameters) > 1 and isinstance(test_input, list):
            if len(input_metadata) > 1 or input_metadata[0] != ast.List.__name__:
                assert namespace[func_name](*test_input) == expected_output
            else:
                assert namespace[func_name](test_input) == expected_output
            return True
        except AssertionError as e:
            return False
        except Exception as e:
            print(f"Could not evaluate TF due to the following error: {e}")
            return False





if __name__ == "__main__":

    x = """from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    stack = []
    current = ""
    
    for char in paren_string:
        if char == "(":
            stack.append("(")
            current += char
        elif char == ")":
            stack.pop()
            current += char
            if not stack:
                result.append(current)
                current = ""
        else:
            current += char
    
    return result

# Example usage
print(separate_paren_groups("(a(b)c) (d(e)f)")) # Output: ["(a(b)c)", "(d(e)f)"]

x = [1, 2, 3]

def add(x,a):
    return x + a

"""

    qn_desc, examples = HumanEvalHelper.process_llm_function_outputs("make_palindrome", x)
    print(examples)