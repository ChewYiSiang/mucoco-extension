import ast
from prediction_inconsistency.utility.database_helper import PredictionInconsistencyHelper
from typing import Any, List, Tuple

class PredictionInconsistencyCruxEvalHelper(PredictionInconsistencyHelper):
    @staticmethod
    def extract_func_name(prog: str):
        """
        Extracts the function name from a given program
        """
        try:
            tree = ast.parse(prog)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and isinstance(node.name, str):
                    return node.name

        except Exception as e:
            raise e
        
    @staticmethod
    def extract_input_metadata(prog: str, test_input: Any):
        tree = ast.parse(prog)        
        class FunctionArgExtractor(ast.NodeVisitor):
            def __init__(self):
                self.arg_names = []
            def visit_FunctionDef(self, node):
                if isinstance(node.args, ast.arguments):
                    self.arg_names = [arg.arg for arg in node.args.args if isinstance(arg, ast.arg)]

        extractor = FunctionArgExtractor()
        extractor.visit(tree)
        
        arg_names = extractor.arg_names
        input_metadata_dict = {}

        if len(arg_names) == 1:
            input_metadata_dict[arg_names[0]] = PredictionInconsistencyCruxEvalHelper.extract_nested_metadata(test_input)
        elif len(arg_names) > 1 and isinstance(test_input,(list, tuple)):
            for idx, arg in enumerate(arg_names):
                input_metadata_dict[arg] = PredictionInconsistencyCruxEvalHelper.extract_nested_metadata(test_input[idx])
        return input_metadata_dict    
    
    @staticmethod
    def extract_nested_metadata(data: Any):
        if isinstance(data, (list, tuple)):
            metadata_list = []
            for d in data:
                metadata_list.append(PredictionInconsistencyCruxEvalHelper.extract_nested_metadata(d))
            return f"{type(data).__name__}({metadata_list})"
        else:
            return type(data).__name__
        