
import regex as re
import sys
import subprocess
from typing import Tuple, Dict, List
from code_generation.utility.database_helper import DatabaseHelper
import unittest
import io
import contextlib
import matplotlib.pyplot as plt
import multiprocessing

class CodeGenerationBigCodeBenchHelper(DatabaseHelper):
    @staticmethod
    def extract_examples(desc: str) -> List[str]:
        pattern = "Example:", "Examples:"
        for p in pattern:
            if p in desc:
                res = desc.split(p)
                processed_res = [r.strip() for r in res if len(r.strip()) > 0]

                if len(processed_res) > 2:
                    processed_res = [''.join(processed_res[:len(processed_res) - 1]), processed_res[-1]]
                
                return processed_res
        else:
            raise ValueError()
        
    @staticmethod
    def split_code_from_instruct_prompt(instruct_prompt: str):
        """
        While the original bigcodebench also includes a code_prompt, this function still helps by splitting the natural language prompt from the code_prompt. 
        """
        pattern = r"```"
        res = re.split(pattern = pattern, string= instruct_prompt)
        processed_res = [r.strip() for r in res if len(r.strip()) > 0]
        if len(processed_res) != 2:
            raise ValueError
        return processed_res
    
    @staticmethod
    def obtain_full_sol(canonical_sol: str, code_instruct: str, test: str):
        full_sol = code_instruct + "\n" + canonical_sol
        # print(full_sol)

        namespace = {}
        tries = 0
        prev_module = None
        while True and tries < 5:
            try:
                exec(full_sol, namespace)
                exec(test, namespace)
                break
            except ModuleNotFoundError as e:
                # Extract the missing module name
                missing_module = str(e).split("'")[1]
                
                if prev_module == missing_module:
                    raise PackageInstallationError(missing_module=missing_module)

                print(f"Module '{missing_module}' not found. Installing...")

                match missing_module.lower():
                    case "sklearn":
                        missing_module = "scikit-learn"
                    case "skimage":
                        missing_module = "scikit-image"
                    case "cv2":
                        missing_module = "opencv-python"
                    case "crypto":
                        missing_module = "pycryptodome"
                    case _:
                        pass
                    
                # Run pip to install the missing module
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", missing_module],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,)


                tries += 1
                
            except Exception as e:
                raise OriginalDBError(e)
        
        TestCasesClass = namespace['TestCases']
        
        # Load all test methods from the unittest.TestCase subclass `TestCasesClass` into a test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCasesClass)

        # Create a StringIO buffer to capture all stdout and stderr outputs during test execution
        f = io.StringIO()

        # Redirect stdout and stderr to the buffer while running the test suite
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            # Run the test suite and capture the result object
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)

            # Close all matplotlib figures in case any were generated during the test
            plt.close('all')

        # If there were any runtime errors (not assertion failures) during the tests, raise a custom exception
        if len(result.errors) > 0:
            raise FullSolutionFailedError(result.errors)


        return full_sol
    
    @staticmethod
    def update_testcase_with_mutated_func_name(
        mutated_func_name: str,
        test_function: str,
        ):
        pattern = r"task_func"
        replacement = mutated_func_name
        return re.sub(pattern, replacement, test_function)
    
    @staticmethod
    def run_llm_answer(
        processed_output: str, 
        test_function: str, 
        func_name: str, 
        mp_queue: multiprocessing.Queue,
        ):
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        
        if func_name != "task_func":
            test_function = CodeGenerationBigCodeBenchHelper.update_testcase_with_mutated_func_name(
                mutated_func_name=func_name,
                test_function=test_function
            )

        namespace = {}

        try:
            exec(processed_output, namespace)
            exec(test_function, namespace)
        except Exception as e:
            mp_queue.put(LLMAnswerFailedError(e))
            return
        TestCasesClass = namespace['TestCases']
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCasesClass)
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
            plt.close('all')  # Close all open figures

        if len(result.errors) > 0:          # indicating that some error has been caught
            mp_queue.put(LLMAnswerFailedError(result.errors))

    @staticmethod
    def check_test_case(
        test_case: str, 
        code_snippet: str, 
        func_name: str
    ):
        if func_name != "task_func":
            test_case = CodeGenerationBigCodeBenchHelper.update_testcase_with_mutated_func_name(
                mutated_func_name=func_name,
                test_function=test_case
            )

        namespace = {}
        exec(code_snippet, namespace)
        exec(test_case, namespace)
        
        TestCasesClass = namespace['TestCases']
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCasesClass)
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
            plt.close('all')  # Close all open figures
        if len(result.errors) > 0:          # indicating that some error has been caught
            return False
        else:
            return True
        

class PackageInstallationError(Exception):
    def __init__(self, missing_module):
        super().__init__(f"Could not install {missing_module} package.") 
    pass

class OriginalDBError(Exception):
    def __init__(self, error):
        super().__init__(f"An error occured with the DB: {type(error), error}") 
    pass

class FullSolutionFailedError(Exception):
    def __init__(self, error):
        super().__init__(f"Full solution failed due to following errors from test case: {error}") 

class LLMAnswerFailedError(Exception):
    def __init__(self, error):
        super().__init__(f"LLM solution failed due to following error: {type(error).__name__ } > {error}") 