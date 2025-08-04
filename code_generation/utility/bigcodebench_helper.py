
import regex as re
import sys
import subprocess
from typing import Tuple, Dict, List
from code_generation.utility.database_helper import DatabaseHelper
import unittest
import io
import contextlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import types

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
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCasesClass)
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
            plt.close('all')  # Close all open figures
        if len(result.errors) > 0:          # indicating that some error has been caught
            raise FullSolutionFailedError(result.errors)

        return full_sol
    
    def check_test_case():
        pass

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
