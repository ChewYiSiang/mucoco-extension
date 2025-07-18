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
            iter_is_func = isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id not in ('enumerate', 'zip')

            # bool indicating if the iterable is a Name E.g.: for i in list
            # First condition pertains to for i in list
            # second condition pertains to different range functions examples such as: 'for i in range(1,10,5):' or 'for j in range(1,10):'
            iter_is_var = (isinstance(node.iter, (ast.Name, ast.Subscript)) and isinstance(node.iter.ctx, ast.Load)) or (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and (node.iter.func.id in ("range", "zip") and len(node.iter.args) > 1))


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

                return [ast.For(
                    target=new_target,
                    iter=new_iter,
                    body=node.body,
                    orelse=node.orelse
                )]
            
            ## Handles cases like: for key in dict.keys() AND for s in string.split()
            ## Transform: for key in dict.keys()
            ## Into: keys0 = list(dict.keys())
            ##       for (loop_var0, key) in enumerate(keys0):
            elif isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
                random_var_name = f"{node.iter.func.attr}{len(self.rename_map)}"
                self.rename_map.add(random_var_name)

                temp_func_node = ast.Call(
                    func = ast.Name(id = 'list'),
                    args = [ast.Call(func = node.iter.func, args = [], keywords=[])],
                    keywords=[]
                )

                new_line = ast.Assign(
                    targets = [ast.Name(id = random_var_name)],
                    value = temp_func_node
                )

                id = f'loop_var{len(self.rename_map)}'
                self.rename_map.add(id)
                tuple_elts = [ast.Name(id = id, ctx = ast.Store()), node.target]
                new_target = ast.Tuple(elts=tuple_elts, ctx=ast.Store())

                new_iter = ast.Call(
                    func=ast.Name(id='enumerate', ctx=ast.Load()),
                    args=[ast.Name(id = random_var_name, ctx=ast.Load())],
                    keywords=[]
                )
                
                return [
                    new_line, 
                    ast.For(
                            target=new_target,
                            iter=new_iter,
                            body=node.body,
                            orelse=node.orelse
                        )
                    ]
            else:                    
                return [node]
        
    class ForToWhileNodeTransformer(ast.NodeTransformer):
        def __init__(self, input_metadata: Dict[str, str]):
            self.var_used = set()
            self.input_metadata = input_metadata

        def find_iteration(self, node):
            
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id != 'zip':
                args = [self.find_arg_iteration(arg) for arg in node.args]
                return args if len(args) > 1 else args[0]
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, list):
                args = [self.find_iteration(arg) for arg in node]
                return args if len(args) > 1 else args[0]
            else:
                return node
            
        def find_arg_iteration(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, list):
                args = [self.find_iteration(arg) for arg in node]
                return args if len(args) > 1 else args[0]
            elif isinstance(node, ast.UnaryOp):
                return ast.literal_eval(ast.unparse(node))
            elif isinstance(node, ast.Constant):
                return node.value
            else:
                return node
            
        def explore_for_loop_target(self, node):
            if isinstance(node, ast.Tuple):
                target = tuple([subnode.id if isinstance(subnode, ast.Name) else subnode for subnode in node.elts ])
                return target
            else:
                raise ValueError("The loop target is not a Tuple, hence indicating that for2enumerate failed to mutate correctly.")

        def visit_For(self, node):
            node = ASTNodeTransformers.ForToEnumerateTransformer().visit(node)
            fixed_nodes = [ast.fix_missing_locations(n) for n in node]
            
            # print(ast.unparse(fixed_nodes[0]))

            node = fixed_nodes[-1]
            self.generic_visit(node)
                
            start = 0               # integer storing the start of the while loop counter
            step = 1                # integer storing the step to increment the counter for each iteration
            increment = True        # bool storing if the step is increasing or decreasing the count

            ## Extracting arg with their respective metadata
            if isinstance(node.iter, ast.Call):
                raw_func_args = node.iter.args
            else:
                raw_func_args = [node.iter]

            target_name, ele = self.explore_for_loop_target(node.target)

            # print(target_name, ele)
            # print(raw_func_args)

            func_args = self.find_iteration(raw_func_args)
            # print(func_args)
            
            ## Updating the start, step, if necessary
            if isinstance(func_args, list):
                if len(func_args) == 3:
                    start, func_args, step = func_args
                elif len(func_args) == 2:
                    start, func_args = func_args

            if step < 0:
                increment = False
            
            ## Setting up the iteration node used in the while loop
            ## The first if condition looks for the following cases:
            ##      ast.BinOp: for i in (3+10)     -> while i < (3+10)
            ##      ast.Call: for i range(len('hello')) -> while i < len('hello')
            ##      ast.Constant: for s in some_string  -> while i < len(some_string)
            if isinstance(func_args, (ast.BinOp, ast.Call, ast.Constant)) :

                ## Filtering out cases where the function call is a zip(func_args).
                ## The approach to converting them is slightly different.
                ##      E.g.: for a1, a2 in zip(arg1, arg2)  ->  while i < min(len(arg1), len(arg2))
                if isinstance(func_args, ast.Call) and isinstance(func_args.func, ast.Name) and func_args.func.id == "zip":
                    len_nodes = []

                    # Creating ast nodes for each len() nodes
                    for arg in func_args.args:
                        new_len_node = ast.Call(
                            func = ast.Name(id = 'len'),
                            args = [arg],
                            keywords=[]
                        )
                        len_nodes.append(new_len_node)

                    # Creating the min() function
                    iter_node = ast.Call(
                        func = ast.Name(id = 'min'),
                        args = len_nodes,
                        keywords=[]
                    )
                    pass

                else:
                    iter_node = func_args
            
            ## The second if condition utilizes the metadata of the function input types.
            ## If the input type is of type 'int' -> while i < func_arg
            elif self.input_metadata.get(func_args, None) == int.__name__:
                iter_node = ast.Name(id = func_args)

            ## Else, all other cases creates the following nodes, such as lists, etc:
            ##  while i < len(func_args)
            else:
                iter_node = ast.Call(
                    func = ast.Name(id = 'len'),
                    args= [(ast.Name(str(func_args)) if not isinstance(func_args, ast.Subscript) else func_args)],           # ast.Subscript will be len(func_args[:-1]), while the rest will be len(func_args)
                    keywords=[]
                )

            ## Setting up the comparison node in the while loop
            #       E.g.: while i < 10:
            while_loop_condition = ast.Compare(
                left = ast.Name(id = target_name, ctx = ast.Store()),
                ops= [ast.Lt()] if increment is True else [ast.Gt()],
                comparators= [iter_node]
            )
            
            ## Establishing the counter for the while loop
            #       E.g.: i = 0
            init_assign = ast.Assign(
                targets=[ast.Name(id=target_name, ctx=ast.Store())],
                value= (
                    ast.Constant(value=start) if isinstance(start, int)                 # if the start is an int like "i = 6" for example
                    else ast.Name(id=start, ctx=ast.Load()) if isinstance(start, str)   # if the start is a variable name such as "i = n"
                    else start                                                          # if the staet is anything else such as "i = 1+2"
                )
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
            if not isinstance(func_args, (list, type(None))) and isinstance(ele, str) and not ele.startswith('loop_idx'):
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

            ## Filters out instances where the original for loop uses a zip() function on the func_args and creates the following
            ## E.g.: for (a1, a2) in zip(arg1, arg2) -> while i < min(len(arg1), len(arg2))
            ##                                              a1 = arg1[i]  <<< this section creates the following lines
            ##                                              a2 = arg2[i]  <<< 
            elif isinstance(func_args, ast.Call) and isinstance(func_args.func, ast.Name) and func_args.func.id == "zip":
                original_variables = ele.elts
                for idx, arg in enumerate(func_args.args):
                    new_assignment_node = ast.Assign(
                        targets = [original_variables[idx]],
                        value = ast.Subscript(
                            value = arg,
                            slice = ast.Name(id = target_name)
                        )
                    )
                    node.body.insert(0, new_assignment_node)
                pass


            node.body.append(assignment_expr)

            
            ## Setting up the while loop node with the new condition and modified body
            while_loop = ast.While(
                test = while_loop_condition,
                body = node.body,
                orelse= node.orelse
            )
            
            ## Returning the counter assignment node and the while loop node
            return [fixed_nodes[:-1], init_assign, while_loop]