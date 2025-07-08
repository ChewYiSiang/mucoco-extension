import ast
from typing import List, Tuple, Callable, Dict
import textwrap
import random
import string
import re
from code_mutation.ast_mutation import ASTNodeTransformers

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
    ) -> Tuple[str, str, str, Dict[str, str]]:
        # 1) Build rename mapping for all identifiers
        rename_map = {}

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
    
        # 3) Apply renamer to the source code
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        
        var_name_transformer = ASTNodeTransformers.VariableNameTransformer(rename_map=rename_map)
        mutated_source = var_name_transformer.visit(tree)
        ast.fix_missing_locations(mutated_source)
        mutated_source = ast.unparse(mutated_source)

        # 4) Apply renamer to the test_case snippet
        mutated_test_case = {}

        for eg in examples:
            test_tree = ast.parse(eg)
            mutated_test_tree = var_name_transformer.visit(test_tree)
            ast.fix_missing_locations(mutated_test_tree)
            mutated_test_case[ast.unparse(mutated_test_tree)] = examples[eg]
        
        # 5) Applying mutation onto question description, should the original function name appear in there.
        for name in rename_map:
            regex_pattern = rf'\b{re.escape(name)}\b'
            qn_desc = re.sub(regex_pattern, rename_map[name], qn_desc)

        return mutated_source, mutated_test_case, qn_desc, rename_map
    
    @staticmethod
    def generate_random_name() -> str:
        length = random.randrange(3, 15)
        alphabet = string.ascii_letters
        return ''.join(random.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def mutate_for_to_enumerate(
        source: str
    ) -> str:
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        mutated_source = ASTNodeTransformers.ForToEnumerateTransformer().visit(tree)
        ast.fix_missing_locations(mutated_source)

        mutated_code = ast.unparse(mutated_source)

        return mutated_code
    
    @staticmethod
    def mutate_for_to_while(source: str):
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        mutated_source = ASTNodeTransformers.ForToWhileNodeTransformer().visit(tree)
        ast.fix_missing_locations(mutated_source)

        mutated_code = ast.unparse(mutated_source)

        return mutated_code


if __name__ == "__main__":
    x = textwrap.dedent("""
        def jo():
            for i in range(10):
                print(i)

        for x in i:
            print(i)
    """)
    x = CodeMutator.mutate_for_to_while(x)
    print(x)
