import ast
from typing import List, Tuple, Callable, Dict
import textwrap
import random
import string

class CodeMutator:
    @classmethod
    def code_masking(original_code : str, mask_type : List[str] = ["var"]) -> str:
        tree = ast.parse()
        for mask in mask_type:
            print(mask)

        ### Will require a ending step where it executes against the check function

    @staticmethod
    def obtain_key_info_from_code(code : str):
        main_func = set()
        func_names = []
        var_names = []
        annotation_names = set()

        try: 
            tree = ast.parse(code)
        except IndentationError:
            code += "\n" + "    pass"
            tree = ast.parse(code)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                main_func.add(node)
        
        for node in main_func:
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.arguments):
                    var_names.extend([n.arg for n in subnode.args])
                
                if isinstance(subnode, ast.FunctionDef):
                    func_names.append(subnode.name)
                
        func_names = list(dict.fromkeys(func_names))
        var_names = list(dict.fromkeys(var_names))
 
        return func_names, var_names
    
    @staticmethod
    def mutate_variable_names( 
        source: str,
        qn_desc: str,
        examples: List[str],
        func_names: List[str],
        mutation_type : str,
        var_names: List[str] = None,
    ) -> Tuple[str, str, str]:
        # 1) Build rename mapping for all identifiers
        rename_map: dict[str, str] = {}

        if mutation_type.strip().lower() == "sequential":
            for idx, name in enumerate(func_names, start=1):
                rename_map[name] = f"generic_function{idx}"
            if var_names:
                for idx, name in enumerate(var_names, start=1):
                    rename_map[name] = f"var{idx}"
        elif mutation_type.strip().lower() == "random":
            all_targets = list(func_names) + (var_names or [])

            for orig in all_targets:
                new_name = CodeMutator.generate_random_name()
                while new_name in rename_map.values():              # ensures that the random name generator does not generate the same name
                    new_name = CodeMutator.generate_random_name()
                rename_map[orig] = new_name
        else:
            raise ValueError("Invalid type of mutation used")
    
        # 2) AST transformer to rename identifiers
        class VariableTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                # rename function definition
                if node.name in rename_map:
                    node.name = rename_map[node.name]
                self.generic_visit(node)
                return node

            def visit_arg(self, node: ast.arg) -> ast.AST:
                # rename function parameters
                if node.arg in rename_map:
                    node.arg = rename_map[node.arg]
                return node

            def visit_Name(self, node: ast.Name) -> ast.AST:
                # rename all identifier usage
                if node.id in rename_map:
                    node.id = rename_map[node.id]
                return node

        # 3) Apply renamer to the source code
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        mutated_source = VariableTransformer().visit(tree)
        ast.fix_missing_locations(mutated_source)
        mutated_source = ast.unparse(mutated_source)

        # 4) Apply renamer to the test_case snippet
        mutated_test_case = {}

        for eg in examples:
            test_tree = ast.parse(eg)
            mutated_test_tree = VariableTransformer().visit(test_tree)
            ast.fix_missing_locations(mutated_test_tree)
            mutated_test_case[ast.unparse(mutated_test_tree)] = examples[eg]
        
        # 5) Applying mutation onto quetion description, should the original function name appear in there.
        for name in rename_map:
            qn_desc = qn_desc.replace(name, rename_map[name])

        return mutated_source, mutated_test_case, qn_desc
    
    @staticmethod
    def generate_random_name() -> str:
        length = random.randrange(3, 15)
        alphabet = string.ascii_letters
        return ''.join(random.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def mutate_for_to_enumerate(
        source: str
    ) -> str:
        class ForToEnumerateTransformer(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)

                # bool value checking if the iterable is a range function E.g.: for i in range(10)
                iter_is_range = isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"

                # bool indicating if the iterable is a Name E.g.: for i in list
                iter_is_var = isinstance(node.iter, ast.Name) and isinstance(node.iter.ctx, ast.Load)

                if iter_is_range or iter_is_var:
                    # Transform: for i in range(...)
                    # Into: for idx, i in enumerate(range(...))

                    new_target = ast.Tuple(elts=[
                        ast.Name(id='idx', ctx=ast.Store()),  # create idx
                        node.target                             # keep original i
                    ], ctx=ast.Store())

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
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        mutated_source = ForToEnumerateTransformer().visit(tree)
        ast.fix_missing_locations(mutated_source)

        mutated_code = ast.unparse(mutated_source)

        return mutated_code


if __name__ == "__main__":
    x = textwrap.dedent("""
    def string_xor(a: str, b: str) -> str:
        def xor(i, j):
            if i == j:
                return '0'
            else:
                return '1'

        return ''.join(xor(x, y) for x, y in zip(a, b))
    """)
    f, c = CodeMutator.obtain_key_info_from_code(x)
    print(f)
    print(c)