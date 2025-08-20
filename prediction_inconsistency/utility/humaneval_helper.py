import ast
from typing import Tuple, List, Dict, Any
from code_generation.utility.humaneval_helper import CodeGenerationHumanEvalHelper
from prediction_inconsistency.utility.database_helper import PredictionInconsistencyHelper

class PredictionInconsistencyHumanEvalHelper(CodeGenerationHumanEvalHelper, PredictionInconsistencyHelper):
    def extract_input_metadata(
            examples: Dict[str, str], 
            qn: str
        ) -> Dict[str, str]:
        """
        This function is used to extract the metadata of inputs to a function.

        This is especially important for determining the way to mutate for for2while mutations

        E.g.: 
            >>> examples = {'largest_divisor(15)': '5'}
            >>> extract_input_metadata(examples, qn) == {15: 'int'}
            True

            'int' is returned as the input is an integer 15.
        """
        example_inputs = list(examples.keys())
        first_example = example_inputs[0]

        input_metadata = []
        metadata_dictionary = {}

        example_tree = ast.parse(first_example)

        for node in example_tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func_input = node.value.args
                processed_inputs = [ast.unparse(n) for n in func_input]

                for i in processed_inputs:
                    try:
                        input_metadata.append(type(eval(i)).__name__)
                    except:
                        continue
        
        try:
            qn_tree = ast.parse(qn)
        except:
            qn += '\n    pass'
            qn_tree=ast.parse(qn)
        
        for node in qn_tree.body:
            if isinstance(node, ast.FunctionDef) and isinstance(node.args, ast.arguments):
                raw_func_args = node.args.args
                func_args = [arg.arg for arg in raw_func_args]

        if len(func_args) == len(input_metadata):
            for idx in range(len(func_args)):
                func_arg = func_args[idx]
                metadata = input_metadata[idx]
                metadata_dictionary[func_arg] = metadata
        else:
            raise ValueError("The number of arguments extracted does not match with the number of metadata extracted")
        
        return metadata_dictionary

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