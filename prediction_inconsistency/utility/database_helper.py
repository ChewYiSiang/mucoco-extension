import inspect
from typing import List, Tuple, Dict, Any
import ast

class PredictionInconsistencyHelper:
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
            sig = inspect.signature(namespace[func_name])
            if test_input is None:
                assert namespace[func_name]() == expected_output
            elif not isinstance(test_input, (int)) and len(sig.parameters) > 1:
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
        
class AST_Helper:

    COMPARE_OP_MAP = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }

    def route_ast_type(code: ast.AST):
        if isinstance(code, ast.Call):
            return AST_Helper.extract_ast_call_args(code)

        elif isinstance(code, ast.Constant):
            return AST_Helper.extract_ast_constant(code)
        
        elif isinstance(code, ast.UnaryOp):
            return AST_Helper.extract_ast_unaryop(code)
        
        elif isinstance(code, ast.List):
            return AST_Helper.extract_ast_list(code)
        
        elif isinstance(code, list):
            res = [AST_Helper.route_ast_type(c) for c in code]
            return res[0] if len(res) == 1 else res
        
        elif isinstance(code, ast.Tuple):
            return AST_Helper.extract_ast_tuple(code)
        
        elif isinstance(code, ast.Dict):
            return AST_Helper.extract_ast_dict(code)
            
        elif isinstance(code, ast.BinOp):
            return AST_Helper.extract_ast_bin_op(code)
        
        elif isinstance(code, ast.Name):
            return AST_Helper.extract_ast_name(code)
        
        elif isinstance(code, ast.operator):
            if isinstance(code, ast.Mult):
                return "*"
            elif isinstance(code, ast.Sub):
                return "-"
            elif isinstance(code, ast.Div):
                return "/"
            elif isinstance(code, ast.Add):
                return "+"
            elif isinstance(code, ast.Pow):
                return "**"
            else:
                raise ValueError(f"A method to process {type(code)} operator type has not been developed")
        
        elif not isinstance(code, ast.AST):
            return code
        
        else:
            raise ValueError(f"A method to process {type(code)} has not been developed")

    def extract_ast_dict(code: ast.Dict) -> Dict[Any, Any]:
        if isinstance(code, ast.Dict):
            dict_values = code.values
            dict_keys = code.keys
            key_val_pairs = zip(dict_keys, dict_values)
            res = {}
            for pair in key_val_pairs:
                res[AST_Helper.route_ast_type(pair[0])] = AST_Helper.route_ast_type(pair[1])
            return res

        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Dict type.")

    def extract_ast_call_args(code: ast.Call) -> Tuple[str, list[str]]:
        if isinstance(code, ast.Call):
            test_args = [AST_Helper.route_ast_type(arg) for arg in code.args]
            if code.func.id not in dir(__builtins__):
                args_meta_data = [type(arg).__name__ for arg in test_args]
                return test_args[0] if len(test_args) == 1 else test_args, args_meta_data[0] if len(args_meta_data) == 1 else args_meta_data
            else:
                return test_args[0] if len(test_args) == 1 else test_args
        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Call type.")
        
    def extract_ast_constant(code: ast.Constant) -> str:
        if isinstance(code, ast.Constant):
            return code.value
        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Constant type.")
    
    def extract_ast_compare(code: ast.Compare) -> Tuple[Any, str | List[str], Any]:
        if isinstance(code, ast.Compare):
            left_side = code.left
            if isinstance(left_side, ast.AST):
                left = AST_Helper.route_ast_type(left_side)

            comparators = AST_Helper.route_ast_type(code.comparators)

            ops = [AST_Helper.COMPARE_OP_MAP[type(op)] for op in code.ops]                    # a list is used here as there could be more than 1 ops
            
            return (
                left, 
                ops[0] if len(ops) == 1 else ops, 
                comparators
            )
        
        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Constant type.")
        
    def extract_ast_tuple(code: ast.Tuple) -> Tuple[Any]:
        if isinstance(code, ast.Tuple):
            elts = code.elts
            return tuple(AST_Helper.route_ast_type(elt) for elt in elts)
        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Tuple type.")
    
    def extract_ast_bin_op(code: ast.BinOp) -> int:
        if isinstance(code, ast.BinOp):
            left = AST_Helper.route_ast_type(code.left)
            right = AST_Helper.route_ast_type(code.right)
            oper = AST_Helper.route_ast_type(code.op)

            def format_val(val):
                return repr(val) if isinstance(val, str) else val
            
            return (eval(f"{format_val(left)} {oper} {format_val(right)}"))
        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Tuple type.")

    def extract_ast_name(code: ast.Name) -> str:
        if isinstance(code, ast.Name):
            return code.id
        else:
            raise ValueError("Incorrect extraction method used, code snippet is not ast.Name type.")

    def extract_ast_unaryop(code: ast.UnaryOp) -> int | float:
        unary_map = {
            ast.USub: "-",
            ast.UAdd: "+",
            ast.Not: "not ",
            ast.Invert: "~"
        }
        if isinstance(code, ast.UnaryOp):
            symbol = unary_map[type(code.op)]
            value = str(code.operand.value)
            try:
                return int(symbol + value)
            except ValueError:
                return float(symbol + value)
            except Exception as e:
                raise ValueError(f"Unable to extract the unaryop due to the following error: {e}")

    def extract_ast_list(code: ast.List) -> List[Any]:
        if isinstance(code, ast.List):
            list_elements = code.elts
            return [AST_Helper.route_ast_type(elt) for elt in list_elements]
        else:
            print(type(code))
            raise ValueError("Incorrect extraction method used, code snippet is not ast.List type.")

def extract_assert_cases(code: str) -> Tuple[int, List, int]:
    test_cases = []             # list storing the test parameters and test outputs for this check function
    num_cases = 0               # integer storing the number of test cases in this check function
    failed_cases = set()        # assert statements that failed to extract, if any
    rejected_cases = 0           # rejected test cases as the assert statements do not check for "=="

    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        for subnode in node.body:       #iterating through eachline node within the check function
            if not isinstance(subnode, ast.Assert):
                continue 

            test_expr = subnode.test
            num_cases+= 1
            test_outputs = None
            test_params = None

            if isinstance(test_expr, ast.Compare):
                ### Extracts test cases such as "assert candidate([1,2,3]) == 3"
                ops_type = test_expr.ops[0]                 # assuming only one operator in the assert case
                if type(ops_type) != ast.Eq:                # test case rejected as it is not check for equivalence
                    rejected_cases += 1
                    num_cases -= 1                          # not collecting comparisons with "<", ">", etc as test cases
                    continue
                else:
                    test_params, test_operators, test_outputs = AST_Helper.extract_ast_compare(test_expr)
            elif isinstance(test_expr, ast.Call):
                ### Extracts test cases such as assert candidate([1,2,3]) 
                test_params = AST_Helper.extract_ast_call_args(test_expr)
                test_outputs = True

            elif isinstance(test_expr, ast.UnaryOp):
                ### Extracts test cases such as assert not candidate([1,2,3])
                op = test_expr.op
                operand = test_expr.operand
                test_params = AST_Helper.route_ast_type(operand)
                if isinstance(op, ast.Not):
                    test_outputs = False 
                else: 
                    failed_cases.add(num_cases-1)
                
            elif isinstance(test_expr, ast.Constant):
                ### Ignores test cases such as assert True
                num_cases -= 1
                continue
            else:
                print("Can't decide: ", type(test_expr))
                continue
                
            test_cases.append((test_params, test_outputs))
                
        
    return num_cases, test_cases, rejected_cases
