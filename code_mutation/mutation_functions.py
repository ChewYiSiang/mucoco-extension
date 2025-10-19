import ast
from typing import List, Tuple, Dict, Any
import multiprocessing
import inspect
import random
import string
import re
import numpy as np
from code_mutation.ast_mutation import ASTNodeHelper
from prediction_inconsistency.utility.humaneval_helper import PredictionInconsistencyHumanEvalHelper
from prediction_inconsistency.utility.cruxeval_helper import PredictionInconsistencyCruxEvalHelper
from mcq_inconsistency.utility.codemmlu_helper import CodeGenerationCodeMMLUHelper

from utility.constants import Mutations, CodeMMLU, MCQInconsistency, CodeGeneration, Benchmarks, Seed

# Declaring mutation names
FOR2WHILE = Mutations.SyntacticMutations.FOR2WHILE
FOR2ENUMERATE = Mutations.SyntacticMutations.FOR2ENUMERATE

RANDOM_MUTATION = Mutations.LexicalMutations.RANDOM
SEQUENTIAL_MUTATION = Mutations.LexicalMutations.SEQUENTIAL
LITERAL_FORMAT = Mutations.LexicalMutations.LITERAL_FORMAT

DEMORGAN = Mutations.LogicalMutations.DEMORGAN
BOOLEAN_LITERAL = Mutations.LogicalMutations.BOOLEAN_LITERAL
COMMUTATIVE_REORDER = Mutations.LogicalMutations.COMMUTATIVE_REORDER
CONSTANT_UNFOLD = Mutations.LogicalMutations.CONSTANT_UNFOLD
CONSTANT_UNFOLD_ADD = Mutations.LogicalMutations.CONSTANT_UNFOLD_ADD
CONSTANT_UNFOLD_MULT = Mutations.LogicalMutations.CONSTANT_UNFOLD_MULT

# Declaring benchmark names
BIGCODEBENCH = Benchmarks.BigCodeBench.NAME
CODEMMLU = Benchmarks.CodeMMLU.NAME
HUMANEVAL = Benchmarks.HumanEval.NAME
CRUXEVAL = Benchmarks.CruxEval.NAME
TURBULENCE = Benchmarks.Turbulence.NAME

# def run_llm_answer(mutated_sol: str, expected_output: Any, func_name: str, mp_queue: multiprocessing.Queue, test_input: Any = 'no_input'):
#         """
#         This function is used to check if an LLM's answer gives the correct output
#         """
#         namespace = {}
#         try:
#             # Execute the mutated code in isolated namespace
#             exec(mutated_sol, namespace)
#             sig = inspect.signature(namespace[func_name])
#             if test_input == 'no_input':
#                 assert namespace[func_name]() == expected_output
#             elif len(sig.parameters) > 1 and isinstance(test_input, (list, tuple)):
#                 assert expected_output ==  namespace[func_name](*test_input)
#             else:
#                 o = namespace[func_name](test_input) 
#                 assert o == expected_output
#         except Exception as e:
#             mp_queue.put(e)


def run_llm_answer(
        prog: str, 
        func_name: str, 
        error_queue: multiprocessing.Queue, 
        ans_queue: multiprocessing.Queue,
        func_input: Any = 'no_input'):
        """
        This function is used to check if an LLM's answer gives the correct output
        """
        
        namespace = {}
        random.seed(Seed.value)

        try:
            # Execute the mutated code in isolated namespace
            exec(prog, namespace)
            sig = inspect.signature(namespace[func_name])
            # Checking if there are any arguments like *args 
            contains_star_arg = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            if not isinstance(func_input, np.matrix) and func_input == 'no_input':
                prog_output = namespace[func_name]()
            elif (len(sig.parameters) > 1 or contains_star_arg) and isinstance(func_input, (list, tuple)):
                prog_output = namespace[func_name](*func_input)
            else:
                prog_output = namespace[func_name](func_input) 

            ans_queue.put(prog_output)

        except Exception as e:
            error_queue.put(e)



class CodeMutator:
    # Main class for applying various types of code mutations while preserving functionality.
    def __init__(self, func_name: str, mutated_dict: Dict[str, Any], benchmark_set: str):
        self.func_name = func_name
        self.mutated_dict = mutated_dict
        self.benchmark_set = benchmark_set

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
    def verify_with_canon_ans(
        func_output: Any, 
        canon_ans: Any,
        ):

        class MatrixNodeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.contains_np_matrix = False
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == np.matrix.__name__:
                    self.contains_np_matrix = True

        def _assert_identical_matrix(matrix_1, matrix_2):
            assert all(np.array_equal(np.array(m1), np.array(m2)) for m1, m2 in zip(matrix_1, matrix_2)) and len(matrix_1) == len(matrix_2)
        
        try:
            if isinstance(canon_ans, (tuple, list)):
                tree = ast.parse(str(canon_ans))
                matrix_visitor = MatrixNodeVisitor()
                matrix_visitor.visit(tree)

                if matrix_visitor.contains_np_matrix == True:
                    _assert_identical_matrix(canon_ans, func_output)
                    return


            if isinstance(canon_ans, list) and isinstance(func_output, list):
                try:
                    canon_ans = sorted(canon_ans, key=lambda x: (type(x).__name__, x))
                    func_output = sorted(func_output, key=lambda x: (type(x).__name__, x))
                    assert canon_ans == func_output
                # a typeerror catch is needed here in the case of sorting None values
                except TypeError:     
                    return sorted(canon_ans, key=str) == sorted(func_output, key=str)

            else:
                assert canon_ans == func_output
        except (ValueError, AssertionError) as e:
            raise AssertionError()
        except (Exception) as e:
            raise e


    def check_solution_validity(
        self,
        program: str,
        output_args: str, 
        input_args: Any = "no_inputs", 
    ):
        timeout = 20
        error_queue = multiprocessing.Queue()
        ans_queue = multiprocessing.Queue()
        if not isinstance(input_args, np.matrix) and input_args == "no_inputs":
            verify_answer_process = multiprocessing.Process(
                target= run_llm_answer, 
                kwargs = {'prog' : program,
                        'func_name': self.func_name,
                        'error_queue' : error_queue,
                        'ans_queue': ans_queue
                        }
                )

        else:
            # print('yar')
            # print(self.func_name)
            # print(program)
            # print(output_args)
            # print(input_args)
            verify_answer_process = multiprocessing.Process(
                target= run_llm_answer, 
                kwargs = {'prog' : program,
                        'func_input': input_args,
                        'func_name': self.func_name,
                        'error_queue' : error_queue,
                        'ans_queue': ans_queue
                        }
                )
            
        verify_answer_process.start()

        verify_answer_process.join(timeout=timeout)
        if verify_answer_process.is_alive():
            verify_answer_process.kill()
            verify_answer_process.join()
            raise RuntimeError()
        if not error_queue.empty():
            e = error_queue.get()
            raise e
        if not ans_queue.empty():
            prog_ans = ans_queue.get()
        else:
            raise ValueError("Function failed to execute or return a result")
            
        ### Answer Verification Step
        CodeMutator.verify_with_canon_ans(func_output=prog_ans, canon_ans=output_args)

        
    def mutate_for_code_generation(
            self,
            mutation_type:str,
            task_set: str,
        ):

        if task_set not in CodeGeneration.BENCHMARKS:
            raise ValueError("Invalid benchmark data used for CodeGeneration testing")
        
        try: 
            tree = ast.parse(self.mutated_dict['question'])
        except IndentationError:
            self.mutated_dict['question'] += "\n" + "    pass"
            tree = ast.parse(self.mutated_dict['question'])
        
        # Sanity check for formatting. 
        sanitised_question = CodeMutator.parse_through_ast(tree)
        tree = ast.parse(sanitised_question)

        try: 
            self.handle_mutation(
                mutation_type=mutation_type,
                task_set=task_set,
                tree = tree,
            )

        except Exception as e:
            raise MutationFailedError(e)
        
        ## note : No post mutation check is conducted as it is assumed that lexical mutations should not impact the canonical solution
    
    def check_mutation_validity(
            self, 
            tree: ast.AST, 
            mutation_type: str, 
            task_set: str, 
            input_args: Any = None,
        ) -> None:
        ## Checking for any valid for loops before syntactic mutations
        if mutation_type in (FOR2WHILE, FOR2ENUMERATE):
            for_loop_checker = ASTNodeHelper.ForLoopDetectorNodeVisitor()
            
            for_loop_checker.visit(tree)
            for_loop_exists = for_loop_checker.for_loop_exists
            enumerate_iterator_exists = for_loop_checker.enumerator_iterator

            if for_loop_exists == False:
                raise NoForLoopError()
            
            elif mutation_type == FOR2ENUMERATE and enumerate_iterator_exists == True:
                raise InvalidIteratorError(mutation_type=mutation_type)

        ## Checking for any valid boolean operations before DeMorgan mutations
        if mutation_type == DEMORGAN:
            boolean_operation_checker = ASTNodeHelper.BooleanOperationDetectorNodeVisitor()
            boolean_operation_checker.visit(tree)
            boolean_operation_exists = boolean_operation_checker.boolean_operation_exists
            if boolean_operation_exists == False:
                raise NoBooleanOperationError()

        ## Checking for any boolean literals before BOOLEAN_LITERAL mutations
        if mutation_type == BOOLEAN_LITERAL:
            boolean_literal_checker = ASTNodeHelper.BooleanLiteralDetectorNodeVisitor()
            boolean_literal_checker.visit(tree)
            boolean_literal_exists = boolean_literal_checker.boolean_literal_exists
            if boolean_literal_exists == False:
                raise NoBooleanLiteralError()

        ## Checking for any commutative operations before COMMUTATIVE_REORDER mutations
        if mutation_type == COMMUTATIVE_REORDER:
            full_sol = self.mutated_dict['full_sol']
            if task_set in (HUMANEVAL, CODEMMLU):
                examples = self.mutated_dict['examples']
                input_metadata = PredictionInconsistencyHumanEvalHelper.extract_input_metadata(examples = examples, qn = full_sol)
            elif task_set in (CRUXEVAL, TURBULENCE):
                input_metadata = PredictionInconsistencyCruxEvalHelper.extract_input_metadata(prog=full_sol, test_input=input_args)
            variable_metadata = CodeMutator.obtain_variable_types(tree, input_metadata)
            merged_metadata = input_metadata | variable_metadata
            
            commutative_operation_checker = ASTNodeHelper.CommutativeOperationDetectorNodeVisitor(metadata_dict=merged_metadata)
            commutative_operation_checker.visit(tree)
            commutative_operation_exists = commutative_operation_checker.commutative_operation_exists
            if commutative_operation_exists == False:
                raise NoCommutativeOperationError()

        ## Checking for any constants to unfold before CONSTANT_UNFOLD mutations
        if mutation_type in (CONSTANT_UNFOLD, CONSTANT_UNFOLD_ADD, CONSTANT_UNFOLD_MULT):
            constant_unfold_checker = ASTNodeHelper.ConstantUnfoldDetectorNodeVisitor()
            constant_unfold_checker.visit(tree)
            constant_unfold_exists = constant_unfold_checker.constant_unfold_exists
            if constant_unfold_exists == False:
                raise NoConstantUnfoldError()


    def mutate_for_mcq_inconsistency(
            self,
            mutation_type:str,
            task_set: str,
            answer: str,
            task_type: str = CodeMMLU.Tasks.CODE_COMPLETION
        ):

        # integer indicating the number of seconds that the program should finish running in
        timeout = 10

        ## Checking the validity of the task_type input
        # If statement checking if the task_type is equal to the code_completion string
        if task_type == CodeMMLU.Tasks.CODE_COMPLETION:
            test_set_helper = CodeGenerationCodeMMLUHelper

        # Elif statement checking if the task_type is not a valid task in CodeMMLU
        elif task_type not in [attr for attr in dir(CodeMMLU.Tasks) if not attr.startswith("_")]:
            # Raising ValueError if so
            raise ValueError("Input task type is invalid")

        ## Checking if the task_set is a benchmark for valid MCQInconsistency testing
        if task_set not in MCQInconsistency.BENCHMARKS:
            # Raising ValueError if the task_set is not adapted for MCQInconsistency testing
            raise ValueError("Invalid benchmark data used for MCQInconsistency testing")

        ## Parsing the original quesiton into an ast tree
        try: 
            tree = ast.parse(self.mutated_dict['full_sol'])
        except IndentationError:
            self.mutated_dict['full_sol'] += "\n" + "    pass"
            tree = ast.parse(self.mutated_dict['full_sol'])

        sanitised_question = CodeMutator.parse_through_ast(tree)
        tree = ast.parse(sanitised_question)

        try: 
            self.check_mutation_validity(tree = tree, mutation_type=mutation_type, task_set = task_set,)
        except Exception as e:
            raise e
        
        ## Handling mutations
        try: 
            self.handle_mutation(
                mutation_type=mutation_type,
                task_set=task_set,
                tree = tree,
            )

            mutated_sol = self.mutated_dict['full_sol']

            func_def, prog_lines = CodeGenerationCodeMMLUHelper.split_mutated_solution(mutated_sol, self.func_name)

            self.mutated_dict['question'] = func_def
            self.mutated_dict['choices'][answer] = prog_lines

            mutated_full_sol = func_def + "\n" + prog_lines

        except Exception as e:
            raise MutationFailedError(e)

        ## Checking if the mutated solution is identical to the original solution
        try:
            if mutation_type in (FOR2ENUMERATE, FOR2WHILE, DEMORGAN, LITERAL_FORMAT):
                assert CodeMutator.standardize_program(mutated_sol) != CodeMutator.standardize_program(sanitised_question)
        except:
            raise IdenticalMutationError(mutation_type=mutation_type)
        
        ## Checking that the mutated solution still passes the check function 
        try:
            multiprocessing_queue = multiprocessing.Queue()

            verify_answer_process = multiprocessing.Process(        
                target= test_set_helper.run_llm_answer,
                args = (mutated_full_sol, self.mutated_dict['check_function'], self.func_name, multiprocessing_queue)
                )
            
            verify_answer_process.start()
            verify_answer_process.join(timeout=timeout)

            if verify_answer_process.is_alive():
                verify_answer_process.kill()
                verify_answer_process.join()
                raise MutationFailedError("The mutated answer took too long to run, which could inidicate some sort of infinite loop")

            if not multiprocessing_queue.empty():
                error = multiprocessing_queue.get()
                raise MutationCheckFailedError(error)

        except Exception as e:
            print(f"DEBUG: Mutation check failed with error: {type(e).__name__}: {e}")
            raise MutationCheckFailedError(e)
        

    
    def handle_mutation(
            self, 
            mutation_type: str,
            task_set: str,
            tree: ast.AST,
            input_args: Any = None,
    ):
        logical_mutations = [getattr(Mutations.LogicalMutations, m) for m in dir(Mutations.LogicalMutations) if not m.startswith("__")]
        lexical_mutations = [getattr(Mutations.LexicalMutations, m) for m in dir(Mutations.LexicalMutations) if not m.startswith("__")]
        syntactic_mutations = [getattr(Mutations.SyntacticMutations, m) for m in dir(Mutations.SyntacticMutations) if not m.startswith("__")]
        try:
            if mutation_type in syntactic_mutations:
                full_sol = self.mutated_dict['question']
                examples = self.mutated_dict.get('examples', None)
                if mutation_type == FOR2WHILE:
                    if task_set in (HUMANEVAL, CODEMMLU):
                        input_metadata = PredictionInconsistencyHumanEvalHelper.extract_input_metadata(examples = examples, qn = full_sol)
                    elif task_set in (CRUXEVAL, TURBULENCE):
                        input_metadata = PredictionInconsistencyCruxEvalHelper.extract_input_metadata(prog=full_sol, test_input=input_args)
                    variable_metadata = CodeMutator.obtain_variable_types(tree, input_metadata)
                    merged_metadata = input_metadata | variable_metadata
                    mutated_sol = CodeMutator.mutate_for_to_while(tree = tree, input_metadata=merged_metadata)  

                elif mutation_type == FOR2ENUMERATE:
                    mutated_sol = CodeMutator.mutate_for_to_enumerate(tree = tree)
                
                self.mutated_dict['question' ] = mutated_sol

            elif mutation_type in logical_mutations:
                full_sol = self.mutated_dict.get('full_sol', None)
                tree = ast.parse(full_sol)
                
                if mutation_type == DEMORGAN:
                    mutated_sol = CodeMutator.mutate_demorgan(tree = tree)
                    
                elif mutation_type == BOOLEAN_LITERAL:
                    mutated_sol = CodeMutator.mutate_boolean_literal(tree = tree)
                
                elif mutation_type == COMMUTATIVE_REORDER:
                    if task_set in (HUMANEVAL, CODEMMLU):
                        examples = self.mutated_dict['examples']
                        input_metadata = PredictionInconsistencyHumanEvalHelper.extract_input_metadata(examples = examples, qn = full_sol)
                    elif task_set in (CRUXEVAL, TURBULENCE):
                        input_metadata = PredictionInconsistencyCruxEvalHelper.extract_input_metadata(prog=full_sol, test_input=input_args)
                    variable_metadata = CodeMutator.obtain_variable_types(tree, input_metadata)
                    merged_metadata = input_metadata | variable_metadata

                    mutated_sol = CodeMutator.mutate_commutative_reorder(tree = tree, metadata_dict=merged_metadata)
                    
                elif mutation_type == CONSTANT_UNFOLD:
                    mutated_sol = CodeMutator.mutate_constant_unfold(tree = tree)
                    
                elif mutation_type == CONSTANT_UNFOLD_ADD:
                    mutated_sol = CodeMutator.mutate_constant_unfold_add(tree = tree)
                    
                elif mutation_type == CONSTANT_UNFOLD_MULT:
                    mutated_sol = CodeMutator.mutate_constant_unfold_mult(tree = tree)
            
            elif mutation_type in lexical_mutations:
            
                if mutation_type in (SEQUENTIAL_MUTATION, RANDOM_MUTATION):
                    func_names, var_names = CodeMutator.obtain_key_info_from_code(tree)

                    mutated_sol = self.mutate_variable_names(
                        func_names=func_names, 
                        mutation_type=mutation_type,
                        var_names=var_names if task_set != TURBULENCE else None,
                    )
                elif mutation_type == LITERAL_FORMAT:
                    mutated_sol = CodeMutator.mutate_literal_format(tree = tree)

            else:
                print("Error at handle_mutation()")
                raise InvalidMutationTypeError(mutation_type= mutation_type, allowed_types= logical_mutations+lexical_mutations+syntactic_mutations)
            
            self.mutated_dict['full_sol'] = mutated_sol

        except Exception as e:
            print(f"Error at handle_mutation(): {type(e)}")
            raise e
        
    
    def mutate_for_prediction_inconsistency_test(
        self,
        mutation_type: str, 
        input_args: Any,
        output_args: Any,
        input_metadata: str,
        task_set: str,
    ):          
        full_sol : str = self.mutated_dict.get('full_sol', None)

        try: 
            tree = ast.parse(full_sol)
        except IndentationError:
            source += "\n" + "    pass"
            tree = ast.parse(full_sol)

        sanitised_question = CodeMutator.parse_through_ast(tree)
        tree = ast.parse(sanitised_question)

        try:
            self.check_mutation_validity(
                tree = tree, 
                mutation_type=mutation_type, 
                task_set = task_set, 
                input_args = input_args
            )
        except Exception as e:
            raise e
        
        try:
            self.handle_mutation(
                mutation_type=mutation_type,
                task_set=task_set,
                tree = tree,
                input_args=input_args
            )   

        except Exception as e:
            raise e
        mutated_sol = self.mutated_dict['full_sol']

        self.mutated_dict['question'] = mutated_sol

        ## Checking if the mutated solution is identical to the original solution
        try:
            if mutation_type in (FOR2ENUMERATE, FOR2WHILE, DEMORGAN, LITERAL_FORMAT, BOOLEAN_LITERAL, COMMUTATIVE_REORDER, CONSTANT_UNFOLD, CONSTANT_UNFOLD_ADD, CONSTANT_UNFOLD_MULT):
                assert CodeMutator.standardize_program(mutated_sol) != CodeMutator.standardize_program(sanitised_question)
        except:
            raise IdenticalMutationError(mutation_type=mutation_type)
        ## Checking if the mutated solution still passes the check function
        try:
            # If statement checking if the there are any inputs for the task function
            if (input_metadata == type(None).__name__):
                self.check_solution_validity(mutated_sol, output_args)
            else:
                self.check_solution_validity(mutated_sol, output_args, input_args)
        except Exception as e:
            print(mutated_sol)
            print(f"DEBUG: Mutation check failed with error: {type(e).__name__}: {e}")
            raise MutationCheckFailedError(e)
    
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

    def mutate_variable_names( 
        self,
        func_names: List[str],
        mutation_type : str,
        var_names: List[str] = None,
    ) -> None:
        # 1) Build rename mapping for all identifiers
        rename_map = {}
        question = self.mutated_dict.get('question', None)
        qn_desc = self.mutated_dict.get('qn_desc', None)
        examples = self.mutated_dict.get('examples', None)
        choices : Dict[str, str] = self.mutated_dict.get('choices', None)
        check_function = self.mutated_dict.get('check_function', None)
        full_sol = self.mutated_dict.get('full_sol', None)

        if mutation_type.strip().lower() == SEQUENTIAL_MUTATION:
            # for loop iterating through each function name
            for idx, name in enumerate(func_names, start=1):
                # updating the corresponding function name in the rename_map dictionary
                rename_map[name] = f"generic_function{idx}"
            
            # if statement checking for any variable names
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
        
        # Updating the function name with the mutated function name
        self.func_name = rename_map[self.func_name]
        
        if question is not None:
            try:
                qn_tree = ast.parse(question)
            except IndentationError:
                question += "\n" + "    pass"
                qn_tree = ast.parse(question)
            
            var_name_transformer = ASTNodeHelper.VariableNameTransformer(rename_map=rename_map)
            mutated_qn = var_name_transformer.visit(qn_tree)
            ast.fix_missing_locations(mutated_qn)
            mutated_qn = ast.unparse(mutated_qn)
            self.mutated_dict['question'] = mutated_qn 

        # 4) Apply renamer to the test_case snippet
        if examples is not None: 
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

            self.mutated_dict['examples'] = mutated_test_case

        # 5) Applying mutation onto question description, should the original function name appear in there.
        if qn_desc is not None:
            for name in rename_map:
                regex_pattern = rf'\b{re.escape(name)}\b'
                qn_desc = re.sub(regex_pattern, rename_map[name], qn_desc)
            self.mutated_dict['qn_desc'] = qn_desc


        # 6) Applying mutation onto question choices, only used for MCQInconsistency mutation cases
        if choices is not None:
            for key, choice in choices.items():
                new_choice = choice
                for name in rename_map:
                    pattern = r'\b{}\b'.format(re.escape(name))  
                    new_choice = re.sub(pattern, rename_map[name], new_choice)
                choices[key] = new_choice
            self.mutated_dict['choices'] = choices
        
        # 7) Applying the mutation onto the check_function
        if check_function is not None and self.benchmark_set not in (HUMANEVAL, CODEMMLU):
            for name in rename_map:
                pattern = r'\b{}\b'.format(re.escape(name))  # safe + whole word
                check_function = re.sub(pattern, rename_map[name], check_function)
            self.mutated_dict['check_function'] = check_function


        # 8) Applying the mutation onto the full solution
        if full_sol is not None:
            full_sol_tree = ast.parse(full_sol)
            mutated_full_sol = var_name_transformer.visit(full_sol_tree)
            ast.fix_missing_locations(mutated_full_sol)
            mutated_full_sol = ast.unparse(mutated_full_sol)
            self.mutated_dict['full_sol'] = mutated_full_sol

        return self.mutated_dict['full_sol']
    
    @staticmethod
    def generate_random_name() -> str:
        length = random.randrange(5, 15)
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
        tree: ast.AST
    ) -> str:

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
        ast.fix_missing_locations(tree)
        original_source = ast.unparse(tree)

        def swap_quotes(match):
            char = match.group(0)
            return "'" if char == '"' else '"'
        
        mutated_code = re.sub(r'["\']', swap_quotes, original_source)
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
    def mutate_commutative_reorder(tree: ast.AST, metadata_dict: Dict) -> str:
        """
        Reorder commutative operations while preserving functionality.
        a + b ↔ b + a, a * b ↔ b * a
        """
        try:
            mutated_source = ASTNodeHelper.CommutativeReorderTransformer(metadata_dict= metadata_dict).visit(tree)
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
    def mutate_constant_unfold_mult(tree) -> str:
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
    def __init__(self, mutation_type):
        message = f"Mutated solution is identical to the original solution after {mutation_type} mutation"
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

class NoBooleanOperationError(Exception):
    """Raised when no boolean operations are in the given program"""
    def __init__(self):
        super().__init__("No valid boolean operations in the given program")

class NoBooleanLiteralError(Exception):
    """Raised when no boolean literals are in the given program"""
    def __init__(self):
        super().__init__("No boolean literals in the given program")

class NoCommutativeOperationError(Exception):
    """Raised when no commutative operations are in the given program"""
    def __init__(self):
        super().__init__("No commutative operations in the given program")

class NoConstantUnfoldError(Exception):
    """Raised when no constants to unfold are in the given program"""
    def __init__(self):
        super().__init__("No constants to unfold in the given program")
class InvalidIteratorError(Exception):
    def __init__(self, mutation_type):
        super().__init__(f"Invalid Iterator for {mutation_type}.")

if __name__ == "__main__":
    print(f"Invalid mutation type was used. The available mutation types are {', '.join(CodeMutator.mutation_types)}")
