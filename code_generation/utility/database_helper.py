import ast
from typing import Tuple
from abc import ABC, abstractmethod
import multiprocessing

class DatabaseHelper(ABC):
    @staticmethod
    def seperate_original_desciptions(prompt: str) -> Tuple[str, str] | None:
        """
        This function takes in the original prompt from the prompt column in the HumanEval Dataset. 
        The prompt should only include the function name, attributes, expected outputs (if provided) and docstrings describing the task.

        For example, from HumanEval/0, the original prompt directly extracted from the HumanEval Dataset is as follows:

            from typing import List

            def has_close_elements(numbers: List[float], threshold: float) -> bool:
                \""" Check if in given list of numbers, are any two numbers closer to each other than
                given threshold.
                >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
                False
                >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
                True
                \"""
        
        This function will then seperate the doc string from the code and return them in a tuple.

        Args:
            prompt (str): original HumanEval "prompt" entry
        
        Returns:
            tuple[str, str]: Tuple containing the seperated program and doc string. The first variable is the program, and the second variable is doc string.

                If no valid program and doc string was extracted, (None, None) is returned instead.

        Raises: 
            Exception: Any errors that may occur from an unsuccessful extraction
        """
        tree = ast.parse(prompt)                                   # obtaining the AST from the prompt
        func_node = None                                                # AST node containing the function

        for idx, node in enumerate(tree.body):                          # for loop iterating through the enumeration of all nodes in the prompt tree
            if isinstance(node, ast.FunctionDef):                       # if statement checking if the node is a function definition type
                func_node = tree.body[idx]                              # setting func node to this node

        tree.body.remove(func_node)                                     # removing the func node, containing the function definition and its accompanying doc string from the AST body
        
        try:
            doc_string = ast.get_docstring(func_node) or None           # obtaining the doc_string from the func_node
            
            if (len(func_node.body) > 0 and                             # checks if there are any code in the func_node
                isinstance(func_node.body[0], ast.Expr) and             # checks that the first item in the func_node.body is indeed an expr, which would correspond with the docstring
                isinstance(func_node.body[0].value, ast.Constant) and   # checks if the value of the first item in the list is a constant
                isinstance(func_node.body[0].value.value, str)):        # checks if the type is a string

                func_node.body.pop(0)                                   # removing the doc string from the function

            tree.body.insert(idx, func_node)                            # inserting the func_node back into the AST body, this time only containing the function definition

            new_prog = ast.unparse(tree)                                # unparsing the new tree into str

            return new_prog, doc_string                                 # returning the new program (without doc strings) and the extracted doc string

        except Exception as e:
            print("Failed due to following error: {e}".format(e = e))
            return None, None                                           # returning a tuple containing None, None
        
    @abstractmethod
    def run_llm_answer(
            processed_output: str, 
            test_function: str, 
            func_name: str, 
            mp_queue: multiprocessing.Queue,
        ):
        pass

    @abstractmethod
    def check_test_case(
            test_case: str, 
            code_snippet: str, 
            func_name: str
        ):
        pass

    
