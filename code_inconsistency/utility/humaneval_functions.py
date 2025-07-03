import ast
from typing import Tuple, List, Dict
from code_generation.utility.humaneval_functions import CodeGenerationHumanEvalHelper


class CodeInconsistencyHumanEvalHelper(CodeGenerationHumanEvalHelper):
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
            if len(input_metadata) > 1 or (input_metadata[0] != ast.List.__name__ and input_metadata[0] != ast.Constant.__name__ ) :
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

    qn_desc, examples = CodeGenerationHumanEvalHelper.process_llm_function_outputs("make_palindrome", x)
    print(examples)