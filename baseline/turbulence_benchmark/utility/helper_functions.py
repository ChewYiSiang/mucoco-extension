import os
import importlib.util
from typing import List, Any
import ast
import re
import multiprocessing as mp
from contextlib import suppress
import numpy as np

def run_tests(
        tests: str, 
        solution: str, 
        test_names: List[str], 
        error_queue: mp.Queue, 
        func_name:str
    ) -> None:
    with suppress(Exception):
        namespace = {}

        exec(tests, namespace)
        exec(solution, namespace)

        for test_name in test_names:
            if test_name != "ignore_warnings":
                try:
                    namespace[test_name]()
                except Exception as e:
                    error_queue.put(e)
                    raise e


class TurbulenceBenchmarkHelper:
    def __init__(self, q_no: int, seed: int):
        self.q_no = q_no
        self.seed = seed

    def return_template_contents(dir):
        with open(dir) as file:
            return file.read()
    
    def run_gen_params(
        self,
        qn_folder_dir: str,
    ) -> List[Any]:
        target_dir = os.path.join(qn_folder_dir, "genparams.py")

        if not os.path.isfile(target_dir):
            raise FileNotFoundError(f"Could not find genparams.py in {qn_folder_dir}")

        spec = importlib.util.spec_from_file_location("gen_params", target_dir)
        func_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(func_module)

        curr_dir = os.getcwd()

        # changing the directory to the qn_folder_dir. Some gen_params look into other files in the qn_folder.
        os.chdir(os.path.join(qn_folder_dir, ".."))
        res = func_module.gen_params(q_no = self.q_no, seed = self.seed)
        os.chdir(curr_dir) 
        return res
    
    def run_input_generator(
        self,
        qn_folder_dir: str,
        gen_params: List,
    ) -> List[Any]:
        target_dir = os.path.join(qn_folder_dir, "gen_function_params.py")

        if not os.path.isfile(target_dir):
            raise FileNotFoundError(f"Could not find gen_function_params.py in {qn_folder_dir}")

        spec = importlib.util.spec_from_file_location("input_generator", target_dir)
        func_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(func_module)

        res = []
        for gen_param in gen_params:
            gen_func_params_res = func_module.input_generator(l = (gen_param), random_seed = self.seed)
            res.append(gen_func_params_res)

        return res
    
    def process_test_cases(
        self,
        test_template: str,
    ) -> str:
        qn_folder_name = f"Q{self.q_no}"
        test_template_lines = test_template.splitlines()

        for idx, line in enumerate(test_template_lines):
            if qn_folder_name in line:
                test_template_lines.pop(idx)
                return '\n'.join(test_template_lines)
        
        return test_template
    
    def obtain_func_name(
        self, 
        sol_template: str,
        qn_txt_template: str,
    ) -> str:
        match = re.search(r"'([^']+)'", qn_txt_template)
        if match:
            func_name = match.group(1)

        sol_tree = ast.parse(sol_template)

        for node in sol_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node.name
        else:
            raise ValueError(f"Could not extract the function name from Q{self.q_no}")
                
    def replace_func_name(
        self,
        tests_template: str,
        func_name: str
    ) -> str:
        regex_pattern = "\$(\d+)"
        res = re.sub(pattern = regex_pattern, repl = func_name, string = tests_template)
        return res

    def run_tests(
        self,
        tests: str,
        solution: str,
        func_name: str,
    ):
        
        class TestNameExtractor(ast.NodeVisitor):
            def __init__(self):
                self.test_names = []

            def visit_FunctionDef(self, node):
                self.test_names.append(node.name)

        test_tree = ast.parse(tests)
        test_name_extractor = TestNameExtractor()
        test_name_extractor.visit(test_tree)
        test_names = test_name_extractor.test_names

        timeout = 30
        error_queue = mp.Queue()
        run_tests_process = mp.Process(
            target= run_tests,
            kwargs={
                "tests": tests,
                "solution": solution,
                "test_names": test_names,
                "error_queue": error_queue,
                "func_name": func_name
            }
        )

        run_tests_process.start()
        run_tests_process.join(timeout=timeout)

        if run_tests_process.is_alive():
            run_tests_process.kill()
            run_tests_process.join()
            raise RuntimeError("Ran for too long.")
        
        if not error_queue.empty():
            e = error_queue.get()
            raise e
        
    
    def process_for_mongo_db_storage(
            self, 
            db_entry: Any
        ):
        if isinstance(db_entry, np.matrix):
            list_entry = db_entry.tolist()
            return {
                "data": list_entry,
                "metadata": np.matrix.__name__
            }
        else:
            return {
                "data": str(db_entry),
                "metadata": type(db_entry).__name__
            }


    
