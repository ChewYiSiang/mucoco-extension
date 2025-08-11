from code_inconsistency.utility.database_helper import CodeInconsistencyHelper
import ast
import re

class ForLoopNodeTransformer(ast.NodeTransformer):
    def __init__(self, target_line_num: int, counter_name: str):
        self.target_line_num = target_line_num
        self.counter_name = counter_name

    def visit_FunctionDef(self, node):
        if isinstance(node.args, ast.arguments):
            self_attr_node = ast.Name(id = "self")
            node.args.args.insert(0, self_attr_node)
        
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        if node.lineno == self.target_line_num:
            counter_node = ast.AugAssign(
                target=ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=self.counter_name,
                    ctx=ast.Store()
                ),
                op=ast.Add(),
                value=ast.Constant(value=1)
            )
            node.body.insert(0, counter_node)
        return node

class CodeInconsistencyCodeSenseHelper(CodeInconsistencyHelper):
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
    def extract_task_input(prog: str):
        prog_lines = prog.splitlines()
        input_line = prog_lines[-1]

        try:
            tree = ast.parse(prog)
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    func_name = node.value.func.id
                    args = []
                    for arg in node.value.args:
                        args.append(eval(ast.unparse(arg)))
                    return func_name, args

        except Exception as e:
            raise e
    
    @staticmethod
    def find_for_loop_target_line_num(prompt: str):
        pattern = r'\bline\s+(\d+)\b'
        line_num = re.search(pattern, prompt, re.IGNORECASE)
        if line_num is not None:
            return int(line_num.group(1))
        else:
            raise ValueError()
        
    @staticmethod
    def modify_full_sol(prog: str, line_num: int):
        counter_name = "counter"

        full_sol = ast.parse(prog)

        # creating the class counter node assignment
        counter_node = ast.Assign(
            targets = [ast.Attribute(
                attr = counter_name, 
                value = ast.Name(id = "self")
                )],
            value = ast.Constant(value=0)
        )

        # creating the function defition node for the counter class
        class_init_node = ast.FunctionDef(
            name = "__init__",
            args = ast.arguments(
                args = [ast.arg(arg="self", annotation=None)],
                posonlyargs=[],
                defaults=[],
                kwonlyargs=[]
            ),
            body = [counter_node],
            decorator_list=[]
            )
        
        for_loop_transformer = ForLoopNodeTransformer(
            target_line_num=line_num, 
            counter_name=counter_name
            )
        
        for_loop_transformer.visit(full_sol)
        ast.fix_missing_locations(full_sol)
        
        # creating the class definition node to house the function
        class_def_node = ast.ClassDef(
            name = "CounterClass",
            body = [class_init_node, full_sol],
            decorator_list=[],
            bases = [],
            keywords = []
        )

        ast.fix_missing_locations(class_def_node)
        new_code = ast.unparse(class_def_node)
        return new_code