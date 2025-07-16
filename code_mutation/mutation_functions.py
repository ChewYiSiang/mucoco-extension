import ast
from typing import List, Tuple, Callable, Dict, Any
import multiprocessing
import inspect
import random
import string
import re
from ast_mutation import ASTNodeTransformers
from code_inconsistency.utility.humaneval_functions import CodeInconsistencyHumanEvalHelper

FOR2WHILE = "for2while"
FOR2ENUMERATE = "for2enumerate"
DEMORGAN = "demorgan"
RANDOM_MUTATION = "random"
SEQUENTIAL_MUTATION = "sequential"

def run_llm_answer(mutated_sol: str, expected_output: str, test_input:Any, func_name: str, mp_queue = multiprocessing.Queue):
        namespace = {}

        try:
            # Execute the mutated code in isolated namespace
            exec(mutated_sol, namespace)
            sig = inspect.signature(namespace[func_name])

            if len(sig.parameters) > 1 and isinstance(test_input, list):
                assert namespace[func_name](*test_input) == expected_output
            else:
                assert namespace[func_name](test_input) == expected_output
        except Exception as e:
            mp_queue.put(e)

class CodeMutator:
    # Main class for applying various types of code mutations while preserving functionality.
    
    mutation_types = [FOR2ENUMERATE, FOR2WHILE, DEMORGAN, RANDOM_MUTATION, SEQUENTIAL_MUTATION]

    @classmethod
    def code_masking(original_code : str, mask_type : List[str] = ["var"]) -> str:
        tree = ast.parse()
        for mask in mask_type:
            print(mask)

        ### Will require a ending step where it executes against the check function

    @staticmethod
    def standardize_program(prog: str) -> str:
        cleaned_lines = [l for l in prog.splitlines() if l != ""]
        for idx, l in enumerate(cleaned_lines):
            cleaned_lines[idx] = l.replace(" ", "")
        return "\n".join(cleaned_lines)

    @staticmethod
    def mutate_for_code_inconsistency_test(
        mutation_type: str | None, 
        full_sol: str,
        examples: Dict[str, str],
        qn_desc: str,
        input_args: Any,
        output_args: Any,
    ) -> str:
        mutated_dict = {
            'full_sol' : full_sol,
            'examples': examples,
            'qn_desc' : qn_desc
        }
        
        if mutation_type == None:
            return mutated_dict
        mutation_type = mutation_type.strip()

        timeout = 5


        example = random.choice(list(examples.keys()))
        func_name = CodeInconsistencyHumanEvalHelper.extract_func_name_from_example(example) 

        try:
            if mutation_type == FOR2WHILE:
                input_metadata = CodeInconsistencyHumanEvalHelper.extract_input_metadata(examples = examples, qn = full_sol)
                mutated_sol = CodeMutator.mutate_for_to_while(source = full_sol, input_metadata=input_metadata)                

            elif mutation_type == FOR2ENUMERATE:
                mutated_sol = CodeMutator.mutate_for_to_enumerate(source = full_sol)
                
            elif mutation_type == DEMORGAN:
                mutated_sol = CodeMutator.mutate_demorgan(source = full_sol)
                
            elif mutation_type == SEQUENTIAL_MUTATION or mutation_type == RANDOM_MUTATION:
                func_names, var_names = CodeMutator.obtain_key_info_from_code(full_sol)
                mutated_sol, examples, qn_desc, mutation_rename_map = CodeMutator.mutate_variable_names(
                            source=full_sol, 
                            qn_desc= qn_desc,
                            examples= examples,
                            func_names=func_names, 
                            mutation_type=mutation_type,
                            var_names=var_names,
                        )
                func_name = mutation_rename_map[func_name]
                mutated_dict['examples'] = examples
                mutated_dict['qn_desc'] = qn_desc            
            else:
                raise InvalidMutationTypeError(mutation_type= mutation_type, allowed_types=CodeMutator.mutation_types)
        except Exception as e:
            raise e
        ## Checking if the mutated solution is identical to the original solution

        # print(mutation_type)
        # print(CodeMutator.standardize_program(mutated_sol))
        # print(CodeMutator.standardize_program(full_sol))
        try:
            if mutation_type in (FOR2ENUMERATE, FOR2WHILE, DEMORGAN):
                assert CodeMutator.standardize_program(mutated_sol) != CodeMutator.standardize_program(full_sol)
        except:

            raise IdenticalMutationError()
        
        
        ## Checking if the mutated solution still passes the check function
        try:
            multiprocessing_queue = multiprocessing.Queue()
            verify_answer_process = multiprocessing.Process(target= run_llm_answer, args = (mutated_sol, output_args, input_args, func_name, multiprocessing_queue))
            verify_answer_process.start()

            verify_answer_process.join(timeout=timeout)
            if verify_answer_process.is_alive():
                verify_answer_process.kill()
                verify_answer_process.join()
                raise RuntimeError()
            
            if not multiprocessing_queue.empty():
                error = multiprocessing_queue.get()
                raise error
            
            mutated_dict['full_sol'] = mutated_sol
        except Exception as e:
            raise MutationCheckFailedError()
        return mutated_dict

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

        if mutation_type.strip().lower() == SEQUENTIAL_MUTATION:
            for idx, name in enumerate(func_names, start=1):
                rename_map[name] = f"generic_function{idx}"
            if var_names:
                for idx, name in enumerate(var_names, start=1):
                    rename_map[name] = f"var{idx}"
        elif mutation_type.strip().lower() == RANDOM_MUTATION:
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
        try: 
            mutated_source = ASTNodeTransformers.ForToEnumerateTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error = e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)

        return mutated_code
    
    @staticmethod
    def mutate_for_to_while(
        source: str, 
        input_metadata: Dict[str, str]
    ) -> str:
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        try: 
            mutated_source = ASTNodeTransformers.ForToWhileNodeTransformer(input_metadata= input_metadata).visit(tree)
        except Exception as e:
            raise MutationFailedError(error = e)

        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_demorgan(
        source: str
    ) -> str:
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        try: 
            mutated_source = ASTNodeTransformers.DeMorganTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error = e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code

class MutationError(Exception):
    """Base class for mutation-related errors."""
    pass

class InvalidMutationTypeError(MutationError):
    """Raised when an unknown mutation type is used."""
    def __init__(self, mutation_type, allowed_types):
        message = f"'{mutation_type}' is not a valid mutation type. Allowed types are: {', '.join(allowed_types)}"
        super().__init__(message)

class IdenticalMutationError(MutationError):
    """Raised when the mutated code is identical to the original."""
    def __init__(self):
        message = "Mutated solution is identical to the original solution."
        super().__init__(message)

class MutationCheckFailedError(MutationError):
    """Raised when the mutated solution does not pass the check function."""
    def __init__(self):
        message = f"Mutated solution did not pass the check function."
        super().__init__(message)

class MutationFailedError(MutationError):
    """Raise when the solution could not be mutated."""
    def __init__(self, error):
        message = f"Solution could not be mutated due to the following error: {type(error)} > {error}"
        super().__init__(message)

if __name__ == "__main__":
    print(f"Invalid mutation type was used. The available mutation types are {', '.join(CodeMutator.mutation_types)}")
