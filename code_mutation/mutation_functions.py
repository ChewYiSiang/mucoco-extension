import ast
from typing import List, Tuple, Dict, Any
import multiprocessing
import inspect
import random
import string
import re
from code_mutation.ast_mutation import ASTNodeHelper
from code_inconsistency.utility.humaneval_helper import CodeInconsistencyHumanEvalHelper
from code_inconsistency.utility.cruxeval_helper import CodeInconsistencyCruxEvalHelper

from utility.constants import SyntacticMutations, LexicalMutations

FOR2WHILE = "for2while"
FOR2ENUMERATE = "for2enumerate"
DEMORGAN = "demorgan"
RANDOM_MUTATION = "random"
SEQUENTIAL_MUTATION = "sequential"
LITERAL_FORMAT = "literal_format"
BOOLEAN_LITERAL = "boolean_literal"
COMMUTATIVE_REORDER = "commutative_reorder"
CONSTANT_UNFOLD = "constant_unfold"
CONSTANT_UNFOLD_ADD = "constant_unfold_add"
CONSTANT_UNFOLD_MULT = "constant_unfold_mult"

def run_llm_answer(mutated_sol: str, expected_output: Any, func_name: str, test_input: Any = 'no_input', mp_queue = multiprocessing.Queue):
        namespace = {}
        try:
            # Execute the mutated code in isolated namespace
            exec(mutated_sol, namespace)
            sig = inspect.signature(namespace[func_name])
            if test_input == 'no_input':
                assert namespace[func_name]() == expected_output
            elif len(sig.parameters) > 1 and isinstance(test_input, (list, tuple)):
                assert expected_output ==  namespace[func_name](*test_input)
            else:
                o = namespace[func_name](test_input) 
                assert o == expected_output
        except Exception as e:
            mp_queue.put(e)

class CodeMutator:
    # Main class for applying various types of code mutations while preserving functionality.
    
    mutation_types = [FOR2ENUMERATE, FOR2WHILE, DEMORGAN, RANDOM_MUTATION, SEQUENTIAL_MUTATION, LITERAL_FORMAT, BOOLEAN_LITERAL, COMMUTATIVE_REORDER, CONSTANT_UNFOLD, CONSTANT_UNFOLD_ADD, CONSTANT_UNFOLD_MULT]

    def __init__(self, func_name: str):
        self.func_name = func_name

    @classmethod
    def code_masking(original_code : str, mask_type : List[str] = ["var"]) -> str:
        tree = ast.parse()
        for mask in mask_type:
            print(mask)

        ### Will require a ending step where it executes against the check function

    @staticmethod
    def extract_func_name_from_source(source_code: str) -> str | None:
        """
        Extract the main function name from source code by finding the first function definition.
        
        Args:
            source_code: The source code to analyze
            
        Returns:
            str: Function name if found, None otherwise
        """
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception as e:
            print(f"DEBUG: Could not parse source code for function name: {e}")
        return None

    @staticmethod
    def standardize_program(prog: str) -> str:
        cleaned_lines = [l for l in prog.splitlines() if l != ""]
        for idx, l in enumerate(cleaned_lines):
            cleaned_lines[idx] = l.replace(" ", "")
        return "\n".join(cleaned_lines)
    
    @staticmethod
    def check_semantic_equivalence(
        original_code: str,
        mutated_code: str,
        input_args: Any,
        func_name: str
    ) -> bool:
        """
        Check if original and mutated code produce the same output for given input.
        This is used for semantic-preserving mutations like DeMorgan transformations.
        
        Args:
            original_code: The original source code
            mutated_code: The mutated source code  
            input_args: The test input arguments
            func_name: The function name to test
            
        Returns:
            bool: True if both codes produce identical results
        """
        try:
            # Execute original code
            orig_namespace = {}
            exec(original_code, orig_namespace)
            
            # Execute mutated code
            mut_namespace = {}
            exec(mutated_code, mut_namespace)
            
            # Get function signatures
            orig_sig = inspect.signature(orig_namespace[func_name])
            mut_sig = inspect.signature(mut_namespace[func_name])
            
            # Call both functions with the same input
            if len(orig_sig.parameters) > 1 and isinstance(input_args, list):
                orig_result = orig_namespace[func_name](*input_args)
                mut_result = mut_namespace[func_name](*input_args)
            else:
                orig_result = orig_namespace[func_name](input_args) 
                mut_result = mut_namespace[func_name](input_args)
            
            return orig_result == mut_result
            
        except Exception as e:
            print(f"DEBUG: Semantic equivalence check failed: {e}")
            return False

    @staticmethod
    def check_solution_validity(
        self,
        program: str,
        output_args: str, 
        input_args: Any = "no_inputs", 
    ):
        timeout = 5
        multiprocessing_queue = multiprocessing.Queue()
        if input_args == "no_inputs":
            # verify_answer_process = multiprocessing.Process(target= run_llm_answer, args = (program, output_args, self.func_name, multiprocessing_queue))
            verify_answer_process = multiprocessing.Process(
                target= run_llm_answer, 
                kwargs = {'mutated_sol' : program,
                        'expected_output': output_args,
                        'func_name': self.func_name,
                        'mp_queue' : multiprocessing_queue
                        }
                )

        else:
            verify_answer_process = multiprocessing.Process(
                target= run_llm_answer, 
                kwargs = {'mutated_sol' : program,
                        'expected_output': output_args,
                        'func_name': self.func_name,
                        'test_input': input_args,
                        'mp_queue' : multiprocessing_queue
                        }
                )

        verify_answer_process.start()

        verify_answer_process.join(timeout=timeout)
        if verify_answer_process.is_alive():
            verify_answer_process.kill()
            verify_answer_process.join()
            raise RuntimeError()
        if not multiprocessing_queue.empty():
            e = multiprocessing_queue.get()
            raise e

    def mutate_for_code_inconsistency_test(
        self,
        mutation_type: str | None, 
        full_sol: str,
        examples: Dict[str, str],
        qn_desc: str,
        input_args: Any,
        output_args: Any,
        input_metadata: str,
        task_set: str
    ) -> str:
        mutated_dict = {
            'full_sol' : full_sol,
            'examples': examples,
            'qn_desc' : qn_desc
        }
        # Removing any whitespaces that could lead to a false equivalence
        mutation_type = mutation_type.strip() if isinstance(mutation_type, str) else mutation_type

        example = random.choice(list(examples.keys()))
        try:
            func_name = CodeInconsistencyHumanEvalHelper.extract_func_name_from_example(example)
        except (ValueError, AttributeError) as e:
            # Fallback: extract function name directly from the source code
            print(f"DEBUG: Could not extract function name from example '{example}': {e}")
            print("DEBUG: Attempting to extract function name from source code...")
            func_name = CodeMutator.extract_func_name_from_source(full_sol)
            if not func_name:
                raise ValueError(f"Could not extract function name from source code or examples")

        try: 
            tree = ast.parse(full_sol)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(full_sol)

        # Pre condition check that checks if a valid for loop exists
        if mutation_type in (FOR2WHILE, FOR2ENUMERATE):
            for_loop_checker = ASTNodeHelper.ForLoopDetectorNodeVisitor()
            for_loop_checker.visit(tree)
            for_loop_exists = for_loop_checker.for_loop_exisits

            if for_loop_exists == False:
                raise NoForLoopError()
        
        try:
            if mutation_type == FOR2WHILE:
                if task_set == "HumanEval":
                    input_metadata = CodeInconsistencyHumanEvalHelper.extract_input_metadata(examples = examples, qn = full_sol)
                elif task_set == "CruxEval":
                    input_metadata = CodeInconsistencyCruxEvalHelper.extract_input_metadata(prog=full_sol, test_input=input_args)
                variable_metadata = CodeMutator.obtain_variable_types(tree, input_metadata)
                merged_metadata = input_metadata | variable_metadata
                mutated_sol = CodeMutator.mutate_for_to_while(tree = tree, input_metadata=merged_metadata)  
                # print(full_sol)
                # print(mutated_sol)
            elif mutation_type == FOR2ENUMERATE:
                mutated_sol = CodeMutator.mutate_for_to_enumerate(tree = tree)
                
            elif mutation_type == DEMORGAN:
                mutated_sol = CodeMutator.mutate_demorgan(source = full_sol)
                
            elif mutation_type == LITERAL_FORMAT:
                mutated_sol = CodeMutator.mutate_literal_format(tree = tree)
                
            elif mutation_type == BOOLEAN_LITERAL:
                mutated_sol = CodeMutator.mutate_boolean_literal(tree = tree)
                
            elif mutation_type == COMMUTATIVE_REORDER:
                mutated_sol = CodeMutator.mutate_commutative_reorder(tree = tree)
                
            elif mutation_type == CONSTANT_UNFOLD:
                mutated_sol = CodeMutator.mutate_constant_unfold(tree = tree)
                
            elif mutation_type == CONSTANT_UNFOLD_ADD:
                mutated_sol = CodeMutator.mutate_constant_unfold_add(tree = tree)
                
            elif mutation_type == CONSTANT_UNFOLD_MULT:
                mutated_sol = CodeMutator.mutate_constant_unfold_mult(tree = tree)
                
            elif mutation_type == SEQUENTIAL_MUTATION or mutation_type == RANDOM_MUTATION:
                func_names, var_names = CodeMutator.obtain_key_info_from_code(tree)
                mutated_sol, examples, qn_desc, mutation_rename_map = CodeMutator.mutate_variable_names(
                            tree=tree, 
                            qn_desc= qn_desc,
                            examples= examples,
                            func_names=func_names, 
                            mutation_type=mutation_type,
                            var_names=var_names,
                        )
                self.func_name = mutation_rename_map[self.func_name]

                mutated_dict['examples'] = examples
                mutated_dict['qn_desc'] = qn_desc       

            elif mutation_type == None:
                mutated_sol = CodeMutator.parse_through_ast(tree)

            else:
                raise InvalidMutationTypeError(mutation_type= mutation_type, allowed_types=CodeMutator.mutation_types)
        except Exception as e:
            raise e

        ## Checking if the mutated solution is identical to the original solution
        try:
            if mutation_type in (FOR2ENUMERATE, FOR2WHILE, DEMORGAN, LITERAL_FORMAT, BOOLEAN_LITERAL, COMMUTATIVE_REORDER, CONSTANT_UNFOLD, CONSTANT_UNFOLD_ADD, CONSTANT_UNFOLD_MULT):
                assert CodeMutator.standardize_program(mutated_sol) != CodeMutator.standardize_program(full_sol)
        except:
            raise IdenticalMutationError()
        ## Checking if the mutated solution still passes the check function
        try:
            # If statement checking if the there are any inputs for the task function
            if (input_metadata == type(None).__name__):
                self.check_solution_validity(mutated_sol, output_args)
            else:
                self.check_solution_validity(mutated_sol, output_args, input_args)

                
            mutated_dict['full_sol'] = mutated_sol
        except Exception as e:
            print(f"DEBUG: Mutation check failed with error: {type(e).__name__}: {e}")
            raise MutationCheckFailedError()
        return mutated_dict
    
    @staticmethod
    def obtain_variable_types(tree: ast.AST, metadata_map: Dict[str, str]) -> Dict[str, str]: 
        """
        This method is used to map variable names to their variable types for ast.Assign nodes.
        
        This is required for for2while mutation as the mutator cannot determine between different data types without the necessary context.

        E.g.: n = 10 
              while i < len(n):            # this line is incorrect and should be while i < n
        
        Hence, this supplements the input_metadata input for the code mutator as it provides the necessary context.

        Args:
            tree (ast.AST): an ast node of any ast.AST type

        Returns: 
            Dict[str, str]: the fully mapped variable dictionary
        """
        node_visitor = ASTNodeHelper.VariableTypeMapperNodeVisitor(metadata_map= metadata_map)
        node_visitor.visit(tree)
        return node_visitor.metadata_map


    @staticmethod
    def obtain_key_info_from_code(prog : ast.Module):
        main_func = set()
        func_names = []
        var_names = []

        for node in prog.body:
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
    def parse_through_ast(
        tree: ast.AST
    ) -> str:
        mutated_source = ASTNodeHelper.DummyTransformer().visit(tree)
        ast.fix_missing_locations(mutated_source)
        mutated_source = ast.unparse(mutated_source)
        return mutated_source

    @staticmethod
    def mutate_variable_names( 
        tree: ast.AST,
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
        
        var_name_transformer = ASTNodeHelper.VariableNameTransformer(rename_map=rename_map)
        mutated_source = var_name_transformer.visit(tree)
        ast.fix_missing_locations(mutated_source)
        mutated_source = ast.unparse(mutated_source)

        # 4) Apply renamer to the test_case snippet
        mutated_test_case = {}

        if isinstance(examples, dict):
            for eg in examples:
                test_tree = ast.parse(eg)
                mutated_test_tree = var_name_transformer.visit(test_tree)
                ast.fix_missing_locations(mutated_test_tree)
                mutated_test_case[ast.unparse(mutated_test_tree)] = examples[eg]
        else:
            # assuming it is BigCodeBench, in which all function names are task_func
            func_name = 'task_func'
            pattern = r'\btask_func\b'
            mutated_test_case = re.sub(pattern, rename_map[func_name], examples)
                
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
        tree: ast.AST
    ) -> str:
        try: 
            mutated_source = ASTNodeHelper.ForToEnumerateTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error = e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_for_to_while(
        tree: ast.AST, 
        input_metadata: Dict[str, str]
    ) -> str:
        try: 
            mutated_source = ASTNodeHelper.ForToWhileNodeTransformer(input_metadata= input_metadata).visit(tree)
        except Exception as e:
            raise MutationFailedError(error = e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)

        return mutated_code
    
    @staticmethod
    def mutate_demorgan(
        source: str
    ) -> str:
        print(f"\n=== DEBUG: ORIGINAL CODE FOR DEMORGAN ===")
        # Print line by line with numbers
        for i, line in enumerate(source.split('\n'), 1):
            print(f"{i:2d}: {line}")
        print("=" * 50)
        
        try: 
            tree = ast.parse(source)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(source)
        try: 
            mutated_source = ASTNodeHelper.DeMorganTransformer().visit(tree)
        except Exception as e:
            print(f"DEBUG: DeMorgan transformation failed: {e}")
            raise MutationFailedError(error = e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        
        print(f"=== DEBUG: MUTATED CODE FOR DEMORGAN ===")
        # Print line by line to avoid truncation
        for i, line in enumerate(mutated_code.split('\n'), 1):
            print(f"{i:2d}: {line}")
        print("=" * 50)
        
        return mutated_code

    @staticmethod
    def mutate_literal_format(tree: ast.AST) -> str:
        """
        Change formatting of string literals while keeping values the same.
        'hello' ↔ "hello"
        """
        try:
            mutated_source = ASTNodeHelper.LiteralFormatTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error=e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_boolean_literal(tree: ast.AST) -> str:
        """
        Change boolean literal representations while keeping logical values the same.
        True ↔ not False, False ↔ not True
        """
        try:
            mutated_source = ASTNodeHelper.BooleanLiteralTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error=e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_commutative_reorder(tree: ast.AST) -> str:
        """
        Reorder commutative operations while preserving functionality.
        a + b ↔ b + a, a * b ↔ b * a
        """
        try:
            mutated_source = ASTNodeHelper.CommutativeReorderTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error=e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_constant_unfold(tree: ast.AST) -> str:
        """
        Unfold constant expressions with random choice (addition/multiplication).
        Falls back to addition if multiplication fails.
        E.g., 10 ↔ 5 + 5 OR 2 * 5
        """
        try:
            mutated_source = ASTNodeHelper.ConstantUnfoldTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error=e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_constant_unfold_add(tree: ast.AST) -> str:
        """
        Unfold constant expressions using addition only.
        E.g., 10 ↔ 5 + 5, 7 ↔ 3 + 4
        """
        try:
            mutated_source = ASTNodeHelper.ConstantUnfoldAddTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error=e)
        
        ast.fix_missing_locations(mutated_source)
        mutated_code = ast.unparse(mutated_source)
        return mutated_code
    
    @staticmethod
    def mutate_constant_unfold_mult(tree: ast.AST) -> str:
        """
        Unfold constant expressions using multiplication only.
        Only transforms if factorization is possible.
        E.g., 10 ↔ 2 * 5, 6 ↔ 2 * 3 (but 7 stays as 7)
        """
        try:
            mutated_source = ASTNodeHelper.ConstantUnfoldMultTransformer().visit(tree)
        except Exception as e:
            raise MutationFailedError(error=e)
        
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
    def __init__(self, error):
        message = f"Mutated solution did not pass the check function  due to the following error: {type(error)} > {error}"
        super().__init__(message)

class MutationFailedError(MutationError):
    """Raise when the solution could not be mutated."""
    def __init__(self, error):
        message = f"Solution could not be mutated due to the following error: {type(error)} > {error}"
        super().__init__(message)

class NoForLoopError(Exception):
    """Raised when no for loops are in the given program"""
    def __init__(self, *args):
        super().__init__("No valid for loops in the given program")

if __name__ == "__main__":
    print(f"Invalid mutation type was used. The available mutation types are {', '.join(CodeMutator.mutation_types)}")
