import ast
from typing import Dict

class ASTNodeTransformers:
    class VariableNameTransformer(ast.NodeTransformer):
        def __init__(self, rename_map: Dict[str, str]):
            self.rename_map = rename_map

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            # rename function definition
            if node.name in self.rename_map:
                node.name = self.rename_map[node.name]
            self.generic_visit(node)
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            # rename function parameters
            if node.arg in self.rename_map:
                node.arg = self.rename_map[node.arg]
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            # rename all identifier usage
            if node.id in self.rename_map:
                node.id = self.rename_map[node.id]
            return node
    
    class ForToEnumerateTransformer(ast.NodeTransformer):
        def __init__(self):
            self.rename_map = set()

        def visit_For(self, node):
            self.generic_visit(node)

            # bool value checking if the iterable is a range function E.g.: for i in range(10)
            iter_is_func = isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id != 'enumerate'

            # bool indicating if the iterable is a Name E.g.: for i in list
            # First condition pertains to for i in list
            # second condition pertains to 'for i in range(1,10,5):' or 'for j in range(1,10):'
            iter_is_var = (isinstance(node.iter, ast.Name) and isinstance(node.iter.ctx, ast.Load)) or (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range" and len(node.iter.args) > 1)


            if iter_is_func or iter_is_var:
                # Transform: for i in range(...)
                # Into: for idx, i in enumerate(range(...))
                id = f'loop_var{len(self.rename_map)}'
                self.rename_map.add(id)

                if iter_is_var:
                    tuple_elts = [ast.Name(id = id, ctx = ast.Store()), node.target]
                else:
                    tuple_elts = [node.target, ast.Name(id = id, ctx = ast.Store())]

                new_target = ast.Tuple(elts=tuple_elts, ctx=ast.Store())

                new_iter = ast.Call(
                    func=ast.Name(id='enumerate', ctx=ast.Load()),
                    args=[node.iter],
                    keywords=[]
                )

                return ast.For(
                    target=new_target,
                    iter=new_iter,
                    body=node.body,
                    orelse=node.orelse
                )
            return node
        
    class ForToWhileNodeTransformer(ast.NodeTransformer):
        def __init__(self, input_metadata: Dict[str, str]):
            self.var_used = set()
            self.input_metadata = input_metadata

        def find_iteration(self, node):
            if isinstance(node, ast.Call):
                args = [self.find_iteration(arg) for arg in node.args]
                return args if len(args) > 1 else args[0]
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, list):
                args = [self.find_iteration(arg) for arg in node]
                return args if len(args) > 1 else args[0]
            else:
                return node

        def explore_for_loop_target(self, node):
            if isinstance(node, ast.Tuple):
                target = tuple([subnode.id for subnode in node.elts])
            else:
                raise ValueError("NOT TUPLEEEE")
            return target

        def visit_For(self, node):
            node = ASTNodeTransformers.ForToEnumerateTransformer().visit(node)
            ast.fix_missing_locations(node)            

            self.generic_visit(node)
                
            start = 0               # integer storing the start of the while loop counter
            step = 1                # integer storing the step to increment the counter for each iteration

            ## Extracting arg with their respective metadata
            raw_func_args = node.iter.args
            
            func_args = self.find_iteration(raw_func_args)
            
            if isinstance(func_args, list):
                if len(func_args) == 3:
                    start, func_args, step = func_args
                elif len(func_args) == 2:
                    start, func_args = func_args

            # func_name, func_args = self.find_iteration(node.iter)
            target_name, ele = self.explore_for_loop_target(node.target)

            ## Setting up the iteration node used in the while loop
            if isinstance(func_args, (ast.BinOp)) :
                iter_node = func_args
            elif self.input_metadata.get(func_args, None) == int.__name__:
                iter_node = ast.Name(id = func_args)
            else:
                iter_node = ast.Call(
                    func = ast.Name(id = 'len'),
                    args= [ast.Name(func_args)],
                    keywords=[]
                )
            ## Setting up the comparison node in the while loop
            #       E.g.: while i < 10:
            while_loop_condition = ast.Compare(
                left = ast.Name(id = target_name, ctx = ast.Store()),
                ops= [ast.Lt()],
                comparators= [iter_node]
            )
            
            ## Establishing the counter for the while loop
            #       E.g.: i = 0
            init_assign = ast.Assign(
                targets=[ast.Name(id=target_name, ctx=ast.Store())],
                value=ast.Constant(value=start) if isinstance(start, int) else start
            )

            ## Incrementing the counter variable in the while loop
            #       E.g.: i += 1
            assignment_expr = ast.AugAssign(
                target = ast.Name(id = target_name, ctx = ast.Store()),
                op = ast.Add(),
                value = ast.Constant(value = step)
            )

            ## Setting up the node for new element assignment
            #       E.g.: new_ele = list[idx]
            if func_args is not None and not isinstance(func_args, list) and not ele.startswith("loop_idx"):
                if isinstance(raw_func_args[0], ast.Call):
                    val = ast.Name(id = target_name)
                else: 
                    val = ast.Subscript(
                            value = raw_func_args[0],
                            slice = ast.Name(id = target_name)
                        )
                
                ele_assignment = ast.Assign(
                    targets=[ast.Name(id = ele)],
                    value = val
                )
                node.body.insert(0, ele_assignment)


            node.body.append(assignment_expr)

            
            ## Setting up the while loop node with the new condition and modified body
            while_loop = ast.While(
                test = while_loop_condition,
                body = node.body,
                orelse= []
            )
            
            ## Returning the counter assignment node and the while loop node
            return [init_assign, while_loop]
            