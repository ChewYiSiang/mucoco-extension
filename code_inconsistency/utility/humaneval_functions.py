import ast
from typing import Tuple, List, Dict, Any
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
            if not isinstance(test_input, int) and len(input_metadata) > 1:
                # print(1)
                # print(namespace[func_name](*test_input), type(namespace[func_name](*test_input)))
                # print(expected_output, type(expected_output))
                assert namespace[func_name](*test_input) == expected_output
            else:
                # print(expected_output, type(expected_output))
                # print(namespace[func_name](test_input), type(namespace[func_name](test_input)))
                assert namespace[func_name](test_input) == expected_output
            return True
        except AssertionError as e:
            return False
        except Exception as e:
            print(f"Could not evaluate TF due to the following error: {e}")
            return False
    
    def extract_input_metadata(
            examples: Dict[str, str], 
            qn: str
        ) -> Dict[str, str]:
        """
        This function is used to extract the metadata of inputs to a function.

        This is especially important for determining the way to mutate for for2while mutations

        E.g.: 
            examples = {'largest_divisor(15)': '5'}
            extract_input_metadata(examples) == ['int']

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
    
    @staticmethod
    def check_database_answer(
        full_sol: str, 
        input_args: Any, 
        input_metadata: List[str], 
        output_args: Any, 
        output_metadata: List[str], 
        examples: Dict[str, str]
    ) -> bool:
        
        random_test_case = list(examples.keys())[0]
        func_name = CodeInconsistencyHumanEvalHelper.extract_func_name_from_example(random_test_case)      

        if output_metadata == type(None).__name__:
            output_metadata = "type(None)"
        if not eval(output_metadata) == str:
            output_args = eval(output_args)

        check_soln_validity = CodeInconsistencyHumanEvalHelper.check_input_output(
            full_sol= full_sol,
            test_input= input_args,
            expected_output= output_args,
            func_name=func_name,
            input_metadata = input_metadata
        )
        if not check_soln_validity:
            return False
        else:
            return True

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