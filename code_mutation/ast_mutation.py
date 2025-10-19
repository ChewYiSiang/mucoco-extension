import ast
import random
from typing import Dict, Tuple
from collections.abc import Iterable
from utility.constants import Seed


class ASTNodeHelper:
    """
    This class stores all ast.NodeTransformers or ast.NodeVisitor classes.

    The ast.NodeTransformer(s) stored under this class includes: 
        - DummyTransformer, VariableNameTransformer, ForToEnumerateTransformer, ForToWhileTransformer, ConditionAugmentationTransformer:

    The ast.NodeVisitor(s) stored under this class includes:
        - VariableTypeMapperNodeVisitor
    """

    class DummyTransformer(ast.NodeTransformer):
        """
        This class is used to standardise a program into AST syntax.

        For example:
            original_prog = "x , y = 10, 11"
            new_prog = DummyTransformer.visit(original_prog)
            ast.fix_missing_locations(new_prog)
            print(ast.unparse(new_prog))
            > x,y = (10, 11)     # brackets are added to encapsulate 10 and 11
        
        Running .visit() with this NodeTransformer is necessary as an additional step to filter out tasks that had no meaningful syntactic changes post mutation.
        """
        pass

    class ForLoopDetectorNodeVisitor(ast.NodeVisitor):
        """
        This NodeVisitor class is used to determine if a valid for loop exists in the input program.

        Note:
            [x for x in list] is not considered a valid for loop. Only traditional for x in list: ... are considered for loops valid for mutation.
        """

        def __init__(self):
            self.for_loop_exists = False
            self.enumerator_iterator = True

        def visit_For(self, node: ast.For):
            self.for_loop_exists = True

            if not (
                isinstance(node.iter, ast.Call) 
                and isinstance(node.iter.func, ast.Name) 
                and node.iter.func.id == "enumerate"
            ):
                self.enumerator_iterator = False
            
            self.generic_visit(node)


    class BooleanOperationDetectorNodeVisitor(ast.NodeVisitor):
        """
        This NodeVisitor class is used to determine if valid boolean operations exist in the input program
        that can be mutated with DeMorgan's laws.

        Detects:
            - BoolOp nodes (and/or operations)
            - UnaryOp with Not applied to boolean operations
        """

        def __init__(self):
            self.boolean_operation_exists = False

        def visit_BoolOp(self, node):
            if isinstance(node.op, (ast.And, ast.Or)):
                self.boolean_operation_exists = True
            self.generic_visit(node)

        def visit_UnaryOp(self, node):
            if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.BoolOp):
                self.boolean_operation_exists = True
            self.generic_visit(node)

    class BooleanLiteralDetectorNodeVisitor(ast.NodeVisitor):
        """
        This NodeVisitor class is used to determine if boolean literals exist in the input program
        that can be transformed (True/False values).

        Detects:
            - Constant nodes with boolean values (True, False)
        """

        def __init__(self):
            self.boolean_literal_exists = False

        def visit_Constant(self, node):
            if isinstance(node.value, bool):
                self.boolean_literal_exists = True

    class CommutativeOperationDetectorNodeVisitor(ast.NodeVisitor):
        """
        This NodeVisitor class is used to determine if commutative operations exist in the input program
        that can be reordered (addition, multiplication).

        Detects:
            - BinOp nodes with commutative operations (Add, Mult)
        """

        def __init__(self, metadata_dict: Dict):
            self.commutative_operation_exists = False
            self.metadata_dict = metadata_dict

        def check_node(self, node: ast.AST) -> bool:
            if isinstance(node, ast.Call):
                return self._check_ast_Function(node)
            elif isinstance(node, ast.Constant):
                return self._check_ast_Constant(node)
            elif isinstance(node, ast.Name):
                return self._check_ast_Name(node)
            else:
                return False

        def _check_ast_Function(self, node: ast.Call) -> bool:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("len","int","float"):
                return True
            return False
        
        def _check_ast_Constant(self, node: ast.Constant) -> bool:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return True
            return False
        
        def _check_ast_Name(self, node: ast.Name) -> bool:
            if isinstance(node, ast.Name) and self.metadata_dict.get(node.id, "no_entry") in (int.__name__, float.__name__):
                return True
            return False

        #TODO: HEREEE

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, (ast.Add, ast.Mult)) and self.commutative_operation_exists != True:
                operands = (node.left, node.right)
                if (
                    any(isinstance(n, (ast.Constant, ast.Call, ast.Name)) for n in operands)
                ):
                    for n in operands:
                        self.commutative_operation_exists = self.check_node(n)
                        if self.commutative_operation_exists == True:
                            return            
            return

    class ConstantUnfoldDetectorNodeVisitor(ast.NodeVisitor):
        """
        This NodeVisitor class is used to determine if integer constants exist in the input program
        that can be unfolded into expressions (e.g., 10 -> 5 + 5).

        Detects:
            - Constant nodes with integer values greater than 1
        """

        def __init__(self):
            self.constant_unfold_exists = False

        def visit_Constant(self, node):
            if isinstance(node.value, int) and node.value > 1:
                self.constant_unfold_exists = True


    class VariableTypeMapperNodeVisitor(ast.NodeVisitor):
        """
        This class is used to map variable names to their respective data types. This class was created to help for2while mutation avoid errors like using len() on an integer variable, which will result in TypeError.

        This is provides NodeTransformers in subsequent mutation steps the necessary contextual metadata for forming the correct mutation types.

        Do note that the metadata_map has to be returned.
        """
        def __init__(self, metadata_map: Dict[str, str]):
            self.metadata_map = metadata_map # initializing the metadata_map

        def obtain_data_type(self, node: ast.AST) -> str | None:
            """
            This method is used to obtain the data type of a given node and serves as a helper function for building the metadata_map. 
            
            Currently, this method only handles into ast.Constant, ast.Call or ast.Name instances. Any other instances will have None returned.

            Args:
                node (ast.AST): the ast node

            Returns:
                str: the data type is returned in string format
                None: returned if node type is out of scope
            """
            ## If statement checking for scenario 1 -> E.g.: i = 10 OR s = "hello"

            if isinstance(node, ast.Constant):
                var_type = type(node.value).__name__ # obtaining the variable type in string format
                return var_type
            
            ## If statement checking for scenario 2 -> E.g.: i = len("hello")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # If statement catching cases like n = len(list) which will return an int.
                    # Note: The logic for 'max' is incorrect here as max can also return a string. It's just used as a placeholder to pass CruxEvalTF521
                    if node.func.id in ("len", "max", ):
                        return int.__name__ # int data type returned as len() always returns an integer
                    
                    # Elif statement catching cases like list(x) or dict(x), and returning the type as itself
                    elif node.func.id in (
                        list.__name__,
                        str.__name__,
                        tuple.__name__,
                        dict.__name__,
                        int.__name__,
                        float.__name__,
                    ):
                        return node.func.id

                # Catching cases like n = list.count("1"), in which the data type for m will be an int
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "count":
                    return int.__name__
                
            ## If statement checking for scenario 3 -> E.g.: i = var1
            #  The if statement also checks if the variable type has already been stored in the metadata_map and retrieves the variable type directly
            elif isinstance(node, ast.Name) and self.metadata_map.get(node.id, 'no_entry') is not 'no_entry':
                stored_metadata = self.metadata_map[node.id]
                data_types = (list.__name__, str.__name__, tuple.__name__, dict.__name__, int.__name__, float.__name__, type(None).__name__)
                if any(data_type for data_type in data_types if data_type in stored_metadata):
                    return stored_metadata
                else:
                    return eval(self.metadata_map[node.id])
                
            elif isinstance(node, ast.BinOp):     
                if isinstance(node.left, (ast.List, ast.Tuple)) or isinstance(node.right, (ast.List, ast.Tuple)):
                    return type(node.left).__name__

                if isinstance(node.right,(ast.Constant)):
                    var_type = type(node.right.value).__name__ # obtaining the variable type in string format
                    return var_type
                
                elif isinstance(node.right, (ast.Name)):
                    existing_var = self.metadata_map.get(node.right.id, None)
                    if existing_var == None:
                        return type(None).__name__
                    else:
                        return existing_var
                    

            elif isinstance(node, ast.IfExp):
                if isinstance(node.body, ast.UnaryOp):
                    return self.obtain_data_type(node.body.operand)
                else:
                    return self.obtain_data_type(node.body)
                
            elif isinstance(node, ast.Subscript):
                if not isinstance(node.value, ast.Name):
                    return type(None).__name__
                
                value = node.value.id
                var = self.metadata_map.get(value, "not_in_metadata")

                if var == "not_in_metadata" or not isinstance(var, Iterable):
                    return type(None).__name__

                if isinstance(node.slice, ast.Constant) and  isinstance(node.slice.value, int):
                    slice = node.slice.value
                    return eval(var)[slice] if not isinstance(var, str) else var[slice]
                
                elif isinstance(node.slice, ast.Slice):                        
                    upper = 0 if node.slice.upper == None else node.slice.upper
                    lower = 0 if node.slice.lower == None else node.slice.lower
                    step = 0 if node.slice.step == None else node.slice.step

                    if any(not isinstance(slice, ast.Constant) for slice in [upper, lower, step]):
                        return type(None).__name__
                    
                    return eval(value)[upper:lower:step]if not isinstance(var, str) else value[upper:lower:step]

                else:
                    pass
            
            ## None returned for nodes out of the scope of this method
            return type(None).__name__

        def visit_Assign(self, node):
            """
            This method visits all nodes that are of type ast.Assign.

            ast.Assign nodes refers to nodes where a variable is assigned a value, E.g.: var1 = 10, var2 = "name", var3 = [1,2,3]

            This method only handles cases where a single variable is assigned at a time and if the variable is named. 

            The variable type is extracted using obtain_data_type() and stored in the metadata map.

            Args:
                node 
            """
            if len (node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                var_type = self.obtain_data_type(node.value)
                self.metadata_map[var_name] = var_type


        def visit_For(self, node):
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
            elif isinstance(node.target, (ast.Tuple, ast.List)) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "enumerate":
                var_name = node.target.elts[0].id
            else:
                return
            
            self.metadata_map[var_name] = self.obtain_data_type(node.iter)
            pass


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
            iter_is_var = (
                (
                    isinstance(node.iter, (ast.Name,)) and 
                    isinstance(node.iter.ctx, ast.Load)
                )
                or (
                    isinstance(node.iter, ast.Call) and 
                    isinstance(node.iter.func, ast.Name) and 
                    node.iter.func.id in {"sorted", "range", "zip", "str", "reversed"}
                )
            )

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
            ## Also handles cases like: for i in ('a', 'b', 'c') and for i in "+"
            elif (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute)) or (isinstance(node.iter, (ast.Tuple, ast.Constant, ast.Subscript, ast.ListComp, ast.List))):
                if isinstance(node.iter, (ast.Tuple, ast.Constant, ast.Subscript, ast.ListComp, ast.List)):
                    random_var_name = f"new_var{len(self.rename_map)}"
                    
                    if isinstance(node.iter, ast.Constant):
                        value = [node.iter]
                    else:
                        value = node.iter

                    new_line = ast.Assign(
                        targets = [ast.Name(id = random_var_name)],
                        value = value
                    )           
                else:
                    random_var_name = f"{node.iter.func.attr}{len(self.rename_map)}"

                    temp_func_node = ast.Call(
                        func = ast.Name(id = 'list'),
                        args = [ast.Call(func = node.iter.func, args = [], keywords=[])],
                        keywords=[]
                    )

                    new_line = ast.Assign(
                        targets = [ast.Name(id = random_var_name)],
                        value = temp_func_node
                    )

                self.rename_map.add(random_var_name)

                new_iter = ast.Call(
                    func=ast.Name(id='enumerate', ctx=ast.Load()),
                    args=[ast.Name(id = random_var_name, ctx=ast.Load())],
                    keywords=[]
                )

                id = f'loop_var{len(self.rename_map)}'
                self.rename_map.add(id)
                tuple_elts = [ast.Name(id = id, ctx = ast.Store()), node.target]
                new_target = ast.Tuple(elts=tuple_elts, ctx=ast.Store())
                
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
            
    class DeMorganTransformer(ast.NodeTransformer):
        def __init__(self):
            pass
        
        def visit_BoolOp(self, node):
            self.generic_visit(node)
            
            # Apply De Morgan's laws:
            # not (A and B) = (not A) or (not B)
            # not (A or B) = (not A) and (not B)
            
            # if there is AND
            # "a and b" creates:
            # BoolOp(op=And(), values=[Name(id='a'), Name(id='b')])
            if isinstance(node.op, ast.And):
                # Transform: A and B -> not ((not A) or (not B))
                negated_values = [self._negate_operand(val) for val in node.values]
                inner_or = ast.BoolOp(op=ast.Or(), values=negated_values)
                return ast.UnaryOp(op=ast.Not(), operand=inner_or)
            
            elif isinstance(node.op, ast.Or):
                # Transform: A or B -> not ((not A) and (not B))
                negated_values = [self._negate_operand(val) for val in node.values]
                inner_and = ast.BoolOp(op=ast.And(), values=negated_values)
                return ast.UnaryOp(op=ast.Not(), operand=inner_and)
            
            return node
        
        def _negate_operand(self, operand):
            """Helper method to properly negate an operand, wrapping in parentheses when needed."""
            # For comparison operations, we need to wrap in parentheses to ensure correct precedence
            if isinstance(operand, ast.Compare):
                # Create: not (operand)
                return ast.UnaryOp(op=ast.Not(), operand=operand)
            else:
                # For other operations, standard negation is sufficient
                return ast.UnaryOp(op=ast.Not(), operand=operand)
        
        def visit_UnaryOp(self, node):
            self.generic_visit(node)
            
            # Handle negated boolean operations
            if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.BoolOp):
                operand = node.operand
                
                if isinstance(operand.op, ast.And):
                    # Transform: not (A and B) -> (not A) or (not B)
                    negated_values = [self._negate_operand(val) for val in operand.values]
                    return ast.BoolOp(op=ast.Or(), values=negated_values)
                
                elif isinstance(operand.op, ast.Or):
                    # Transform: not (A or B) -> (not A) and (not B)
                    negated_values = [self._negate_operand(val) for val in operand.values]
                    return ast.BoolOp(op=ast.And(), values=negated_values)
            
            # Handle double negation: not not X -> X
            elif isinstance(node.op, ast.Not) and isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.Not):
                return node.operand.operand
                
            return node
        
    class ForToWhileNodeTransformer(ast.NodeTransformer):
        def __init__(self, input_metadata: Dict[str, str]):
            self.input_metadata = input_metadata
            self.target_name = None

        def find_iteration(self, node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in ('zip', 'str', 'reversed'):
                args = [self._find_arg_iteration(arg) for arg in node.args]
                return args if len(args) > 1 else args[0]
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, list):
                args = [self.find_iteration(arg) for arg in node]
                return args if len(args) > 1 else args[0]
            elif isinstance(node, ast.Tuple):
                res = []
                for arg in node.elts:
                    if isinstance(arg, ast.Name):
                        res.append(arg.id)
                    elif isinstance(arg, ast.Constant):
                        res.append(arg.value)
                return res
            elif isinstance(node, ast.Constant):
                return node.value
            else:
                return node
                
        def _find_arg_iteration(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, list):
                args = [self.find_iteration(arg) for arg in node]
                return args if len(args) > 1 else args[0]
            elif isinstance(node, ast.UnaryOp):
                try:
                    return ast.literal_eval(ast.unparse(node))
                except:
                    return node
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Call):
                args = [self._find_arg_iteration(arg) for arg in node.args]
                return args if len(args) > 1 else args[0]
            else:
                return node
            
        def explore_for_loop_target(self, node):
            if isinstance(node, ast.Tuple):
                target = tuple([subnode.id if isinstance(subnode, ast.Name) else subnode for subnode in node.elts ])
                return target
            else:
                raise ValueError("The loop target is not a Tuple, hence indicating that for2enumerate failed to mutate correctly.")
            
        def visit_Continue(self, node):
            if self.target_name != None:
                new_increment_line = ast.AugAssign(
                    target = ast.Name(id = self.target_name, ctx = ast.Store()),
                    op = ast.Add(),
                    value = ast.Constant(value = 1)
                )
                return [new_increment_line, node]
            else:
                return ast.Pass()
            
        # def visit_Expr(self, node):
        #     # Check if the Expr is a call to pop/remove
        #     if (
        #         isinstance(node.value, ast.Call) and 
        #         isinstance(node.value.func, ast.Attribute) and
        #         self.target_name != None
        #     ):
        #         if node.value.func.attr in ("pop", "remove"):
        #             # Create the decrement line
        #             decrement = ast.AugAssign(
        #                 target=ast.Name(id=self.target_name, ctx=ast.Store()),
        #                 op=ast.Add(),
        #                 value=ast.Constant(value=1)
        #             )
        #             return [node, decrement]  # Return both original expr and new line
        #         elif node.value.func.attr in ('insert'):
        #             increment = ast.AugAssign(
        #                 target=ast.Name(id=self.target_name, ctx=ast.Store()),
        #                 op=ast.Sub(),
        #                 value=ast.Constant(value=1)
        #             )
        #             return [node, increment]  # Return both original expr and new line
        #         else:
        #             return node
        #     return node


        def visit_For(self, node):
            node = ASTNodeHelper.ForToEnumerateTransformer().visit(node)
            fixed_nodes = [ast.fix_missing_locations(n) for n in node]
            
            node = fixed_nodes.pop()

            start = 0               # integer storing the start of the while loop counter
            step = 1                # integer storing the step to increment the counter for each iteration
            increment = True        # bool storing if the step is increasing or decreasing the count

            ## Extracting arg with their respective metadata
            if isinstance(node.iter, ast.Call):
                raw_func_args = node.iter.args
            else:
                raw_func_args = [node.iter]
            target_name, ele = self.explore_for_loop_target(node.target)

            # storing the name of the counter in this instance
            self.target_name = target_name
            func_args = self.find_iteration(raw_func_args)


            ## Updating the start, step, if necessary
            if isinstance(func_args, list):
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'enumerate' and len(node.iter.args) > 1:    # handling edge cases like for idx, ele in enumerate(list, 1)
                    func_args, start = node.iter.args
                elif len(func_args) == 3:
                    start, func_args, step = func_args
                elif len(func_args) == 2:
                    start, func_args = func_args

            
            # if the step is less than 0, this means that the counter is decreasing with each step and hence increment is false
            if step < 0:
                increment = False
                
            ## Establishing the counter for the while loop
            #       E.g.: i = 0
            init_assign = ast.Assign(
                targets=[ast.Name(id=target_name, ctx=ast.Store())],
                value= (
                    ast.Constant(value=start) if isinstance(start, int)                 # if the start is an int like "i = 6" for example
                    else ast.Name(id=start, ctx=ast.Load()) if isinstance(start, str)   # if the start is a variable name such as "i = n"
                    else start                                                          # if the start is anything else such as "i = 1+2"
                )
            )

            ast.fix_missing_locations(init_assign)
            func_arg_type = self.input_metadata.get(func_args, None)

            if isinstance(func_arg_type, str) and func_arg_type.startswith("list"):
                try:
                    func_arg_type = type(eval(func_arg_type)).__name__
                except:
                    pass

            # updating the input_metadata dict with the new variable assignment
            self.input_metadata[target_name] = int.__name__

            ## Setting up the iteration node used in the while loop
            ## The first if condition looks for the following cases:
            ##      ast.BinOp: for i in (3+10)     -> while i < (3+10)
            ##      ast.Call: for i in range(len('hello')) -> while i < len('hello')
            ##      ast.Constant: for s in some_string  -> while i < len(some_string)
            ## Do note that only ast.Call on 'range' functions are rejected here.

            if isinstance(func_args, ast.Call):
                if isinstance(func_args.func, ast.Name) and func_args.func.id == 'zip':
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

                ## Elif statement checking for instances like: for idx, var in reversed(list)
                elif isinstance(func_args.func, ast.Name) and func_args.func.id in ("len", "reversed"):
                    new_var_name = f"length_var{len(self.input_metadata)}"
                    self.input_metadata[new_var_name] = int.__name__
                    to_add = []

                    ## Handles cases where it is iterating through reversed(list)
                    if func_args.func.id in ("reversed"):
                        new_reversed_var_name = ast.Name(f"new_reversed_var{len(self.input_metadata)}")
                        arg = func_args.args[0]
                        # arg_name = self.find_iteration(func_args.args[0])
                        # func_args = arg_name = ast.Name(arg_name) if isinstance(arg_name, str) else arg_name

                        subscript_node = ast.Subscript(
                            # value = ast.Name(id = arg_name) if isinstance(arg_name, str) else arg_name,
                            value= arg,
                            slice = ast.Slice(
                                lower = None,
                                upper = None,
                                step = ast.UnaryOp(op = ast.USub(), operand=ast.Constant(value=1))
                            )
                        )

                        new_var = ast.Assign(
                            targets=[new_reversed_var_name],
                            value = subscript_node
                        )

                        to_add.append(new_var)

                        new_line = ast.Assign(
                            targets = [ast.Name(id = new_var_name)],
                            value = ast.Call(
                                func = ast.Name(id = 'len'),
                                args = [new_reversed_var_name],
                                keywords=[]
                            )
                        )

                        func_args = new_reversed_var_name

                    else:
                        # Assigning the length of the variable to new_var_name
                        new_line = ast.Assign(
                            targets = [ast.Name(id = new_var_name)],
                            value = func_args
                        )
                    to_add.append(new_line)
                    to_add = [ast.fix_missing_locations(node) for node in to_add]
                    fixed_nodes.extend(to_add)

                    iter_node = ast.Name(id = new_var_name)

                elif (isinstance(func_args.func, ast.Name) and func_args.func.id in ("range", "str")) or isinstance(func_args.func, ast.Attribute):
                    iter_node = ast.Call(
                        func = ast.Name(id = 'len'),
                        args= [func_args],           # ast.Subscript will be len(func_args[:-1]), while the rest will be len(func_args)
                        keywords=[]
                    )

                else: 
                    iter_node = func_args

            elif isinstance(func_args, (ast.BinOp, ast.Constant)):
                iter_node = func_args
            
            ## The second if condition utilizes the metadata of the function input types anf func_args is a declared variable name of int type.
            ## If the input type is of type 'int' -> while i < func_arg
            elif self.input_metadata.get(func_args, None) == int.__name__:
                iter_node = ast.Name(id = func_args)

            ## The third if condition catches cases where func_args is an integer
            ## E.g.: while i < 100
            elif isinstance(func_args, int):
                iter_node = ast.Constant(value = func_args)

            ## Else, all other cases creates the following nodes, such as lists, etc:
            ##  while i < len(arg_length)
            elif func_arg_type and func_arg_type in (list.__name__, dict.__name__):
                func_arg_len = f"{func_args}_len"
                new_assignment_node = ast.Assign(
                    targets = [ast.Name(id = func_arg_len)],
                    value = ast.Call(
                        func = ast.Name(id = 'len'),
                        args = [ast.Name(id = func_args)],
                        keywords=[]
                    )
                )
                ast.fix_missing_locations(new_assignment_node)
                fixed_nodes.append(new_assignment_node)

                iter_node = ast.Name(id = func_arg_len, ctx = ast.Load())

            else:                            
                iter_node = ast.Call(
                        func = ast.Name(id = 'len'),
                        args= [(ast.Name(str(func_args)) if not isinstance(func_args, (ast.Subscript, ast.Call, ast.Name)) else func_args)],           # ast.Subscript will be len(func_args[:-1]), while the rest will be len(func_args)
                        keywords=[]
                )


            ## Setting up the comparison node in the while loop
            #       E.g.: while i < 10:
            while_loop_condition = ast.Compare(
                left = ast.Name(id = target_name, ctx = ast.Store()),
                ops= [ast.Lt()] if increment is True else [ast.Gt()],
                comparators= [iter_node]
            )

            ## Incrementing the counter variable in the while loop
            #       E.g.: i += 1
            assignment_expr = ast.AugAssign(
                target = ast.Name(id = target_name, ctx = ast.Store()),
                op = ast.Add(),
                value = ast.Constant(value = step)
            )

            node.body.append(assignment_expr)

            ## Setting up the node for new element assignment
            #       E.g.: new_ele = list[idx]
            if not isinstance(func_args, (list, type(None))) and isinstance(ele, str) and not ele.startswith('loop_var'):
                if isinstance(raw_func_args[0], ast.Call) and isinstance(raw_func_args[0].func, ast.Name) and raw_func_args[0].func.id not in ("str", 'reversed'):
                    val = ast.Name(id = target_name)

                    if self.input_metadata.get(target_name, None) is not None:
                        self.input_metadata[ele] = self.input_metadata[target_name]

                ## Catches cases like dict.values() or dict.keys()
                #  for val in dict.values() -> val = list(dict.values())[idx]
                elif isinstance(func_args, ast.Call) and isinstance(func_args.func, ast.Attribute):
                    subscript_value_node = ast.Call(
                        func = ast.Name(id = "list"),
                        args = [func_args],
                        keywords=[]
                    )

                    val = ast.Subscript(
                            value = subscript_value_node,
                            slice = ast.Name(id = target_name)
                        )
                
                ## Handles cases iterating through a subscript
                #  for val in list[i:] -> val = list[i:][idx]
                elif isinstance(func_args, ast.Subscript):
                    val = func_args
                
                ## Handles cases where func_args is a named variable
                elif isinstance(func_args, ast.Name):
                    val = ast.Subscript(
                        value = func_args,
                        slice = ast.Name(id = target_name)
                    )

                # Handles cases where the func_args is a function call but it is not reversed
                elif (isinstance(raw_func_args[0], ast.Call) and isinstance(raw_func_args[0].func, ast.Name) and raw_func_args[0].func.id != 'reversed'):
                    val = ast.Subscript(
                            value = raw_func_args[0],
                            slice = ast.Name(id = target_name)
                        )
                    
                # Handles cases where the func_arg is a dict, in which we won't be able to do dict[idx].
                # Instead, we will need to do list(dict.keys())[idx]
                elif func_arg_type == dict.__name__:
                    value = ast.Call(
                        func=ast.Name(id='list', ctx=ast.Load()),
                        args=[
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id=func_args, ctx=ast.Load()),
                                    attr='keys',
                                    ctx=ast.Load()
                                ),
                                args=[],
                                keywords=[]
                            )
                        ],
                        keywords=[]
                    )

                    val = ast.Subscript(
                            value = value,
                            slice = ast.Name(id = target_name)
                        )

                else: 
                    val = ast.Subscript(
                            value = ast.Name(id = func_args),
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
            elif isinstance(func_args, ast.Call) and isinstance(func_args.func, ast.Name) and func_args.func.id in( "zip"):
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
            
            ## Handles scenarios where the for loop target consisted of nested tuples
            ## E.g.:   loop_idx0, (x,y) for enumerate(list): -> while loop_idx0 < len(list):
            ##                                                      x,y = list[loop_idx0]
            elif isinstance(ele, ast.Tuple):
                original_variables = ele.elts
                node.body.insert(0, ast.Assign(
                    targets=[ast.Tuple(elts = ele.elts)],
                    value = ast.Subscript(value = ast.Name(id = func_args), slice = ast.Name(id = target_name))
                ))

            ## Setting up the while loop node with the new condition and modified body
            while_loop = ast.While(
                test = while_loop_condition,
                body = [self.visit(n) for n in node.body],
                orelse= [self.visit(n) for n in node.orelse]
            )
            ast.fix_missing_locations(while_loop)

            ## Returning the counter assignment node and the while loop node
            return [
                fixed_nodes, 
                init_assign, 
                while_loop
            ]

    class LiteralFormatTransformer(ast.NodeTransformer):
        """
        Change formatting of string literals while keeping values the same.
        'hello' ↔ "hello"
        """
        def visit_Constant(self, node):
            self.generic_visit(node)
            
            if isinstance(node.value, str):
                # Change quote style while keeping the string value the same
                # This doesn't actually change the AST since Python normalizes quotes,
                # but we can simulate the effect by toggling a preference
                return node
            
            return node
    
    class BooleanLiteralTransformer(ast.NodeTransformer):
        """
        Change boolean literal representations while keeping logical values the same.
        True ↔ not False, False ↔ not True
        """
        def visit_Constant(self, node):
            self.generic_visit(node)
            
            if isinstance(node.value, bool):
                if node.value is True:
                    # Transform True -> not False
                    return ast.UnaryOp(
                        op=ast.Not(),
                        operand=ast.Constant(value=False)
                    )
                elif node.value is False:
                    # Transform False -> not True  
                    return ast.UnaryOp(
                        op=ast.Not(),
                        operand=ast.Constant(value=True)
                    )
            
            return node
    
    class CommutativeReorderTransformer(ast.NodeTransformer):
        """
        Reorder commutative operations while preserving functionality.
        a + b ↔ b + a, a * b ↔ b * a
        """
        def __init__(self, metadata_dict: Dict):
            self.metadata_dict = metadata_dict

        def visit_BinOp(self, node):
            self.generic_visit(node)
            node_visitor = ASTNodeHelper.CommutativeOperationDetectorNodeVisitor(metadata_dict=self.metadata_dict)
            if isinstance(node.op, (ast.Add, ast.Mult)):
                operands = (node.left, node.right)
                if (
                    any(isinstance(n, (ast.Constant, ast.Call, ast.Name)) for n in operands)
                ):
                    for n in operands:
                        if node_visitor.check_node(n) == True:
                            # Swap left and right operands
                            return ast.BinOp(
                                left=node.right,
                                op=node.op,
                                right=node.left
                            )                
            
            return node
    
    class ConstantUnfoldTransformer(ast.NodeTransformer):
        """
        Unfold constant expressions with random choice and fallback.
        E.g., 10 ↔ 5 + 5 OR 2 * 5 (random, falls back to addition)
        """
        @staticmethod
        def _unfold_addition(value):
            """Unfold using addition: n -> a + b where a + b = n"""
            # Handle special cases for small numbers
            if value == 0:
                return ast.BinOp(
                    left=ast.Constant(value=0),
                    op=ast.Add(),
                    right=ast.Constant(value=0)
                )
            elif value == 1:
                return ast.BinOp(
                    left=ast.Constant(value=0),
                    op=ast.Add(),
                    right=ast.Constant(value=1)
                )
            elif value == -1:
                return ast.BinOp(
                    left=ast.Constant(value=0),
                    op=ast.Add(),
                    right=ast.Constant(value=-1)
                )
            else:
                # General case: split number in half
                half = value // 2
                remainder = value - half
                return ast.BinOp(
                    left=ast.Constant(value=half),
                    op=ast.Add(),
                    right=ast.Constant(value=remainder)
                )
        
        @staticmethod
        def _unfold_multiplication(value):
            """Unfold using multiplication: n -> a * b where a * b = n"""
            
            # Handle factorization for positive values
            abs_value = abs(value)
            if abs_value >= 2:
                # Find factors for multiplication
                for factor in range(2, min(abs_value, 10)):
                    if abs_value % factor == 0:
                        other_factor = abs_value // factor
                        
                        # For negative numbers, make the first factor negative
                        if value < 0:
                            factor = -factor
                        
                        return ast.BinOp(
                            left=ast.Constant(value=factor),
                            op=ast.Mult(),
                            right=ast.Constant(value=other_factor)
                        )
            
            # If no factors found or small number, multiply by 1
            return ast.BinOp(
                left=ast.Constant(value=value),
                op=ast.Mult(),
                right=ast.Constant(value=1)
            )
        
        def visit_Constant(self, node):
            self.generic_visit(node)
            
            if isinstance(node.value, int) and node.value > 1:
                # Randomly choose how to unfold the constant
                random.seed(Seed.value)
                unfold_type = random.choice(['add', 'mult'])
                
                if unfold_type == 'add':
                    return self._unfold_addition(node.value)
                elif unfold_type == 'mult':
                    # Try multiplication first
                    mult_result = self._unfold_multiplication(node.value)
                    if mult_result is not None:
                        return mult_result
                    else:
                        # Fallback to addition
                        return self._unfold_addition(node.value)
            
            return node
    
    class ConstantUnfoldAddTransformer(ast.NodeTransformer):
        """
        Unfold constant expressions using addition only.
        E.g., 10 → 5 + 5, 7 → 3 + 4, 1 → 0 + 1, 0 → 0 + 0
        """
        def visit_Constant(self, node):
            self.generic_visit(node)
            
            if isinstance(node.value, int):
                # Handle special cases for small numbers
                if node.value == 0:
                    return ast.BinOp(
                        left=ast.Constant(value=0),
                        op=ast.Add(),
                        right=ast.Constant(value=0)
                    )
                elif node.value == 1:
                    return ast.BinOp(
                        left=ast.Constant(value=0),
                        op=ast.Add(),
                        right=ast.Constant(value=1)
                    )
                elif node.value == -1:
                    return ast.BinOp(
                        left=ast.Constant(value=0),
                        op=ast.Add(),
                        right=ast.Constant(value=-1)
                    )
                else:
                    # General case: split number in half
                    half = node.value // 2
                    remainder = node.value - half
                    return ast.BinOp(
                        left=ast.Constant(value=half),
                        op=ast.Add(),
                        right=ast.Constant(value=remainder)
                    )
            
            return node
    
    class ConstantUnfoldMultTransformer(ast.NodeTransformer):
        """
        Unfold constant expressions using multiplication only.
        Tries factorization first, fallback to multiplication by 1 for primes.
        E.g., 10 → 2 * 5, 6 → 2 * 3, 7 → 7 * 1, 1 → 1 * 1
        """
        def visit_Constant(self, node):
            self.generic_visit(node)
            
            if isinstance(node.value, int):
                # Handle factorization for positive values
                abs_value = abs(node.value)
                if abs_value >= 2:
                    # First try to find factors for meaningful factorization
                    for factor in range(2, min(abs_value, 10)):
                        if abs_value % factor == 0:
                            other_factor = abs_value // factor
                            
                            # For negative numbers, make the first factor negative
                            if node.value < 0:
                                factor = -factor
                            
                            return ast.BinOp(
                                left=ast.Constant(value=factor),
                                op=ast.Mult(),
                                right=ast.Constant(value=other_factor)
                            )
                
                # If no factors found or small number, multiply by 1
                return ast.BinOp(
                    left=ast.Constant(value=node.value),
                    op=ast.Mult(),
                    right=ast.Constant(value=1)
                )

            
            return node
