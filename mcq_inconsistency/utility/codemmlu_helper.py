from prediction_inconsistency.utility.database_helper import PredictionInconsistencyHelper
from prediction_inconsistency.utility.humaneval_helper import PredictionInconsistencyHumanEvalHelper
import textwrap
from typing import Dict, Tuple
import string
import ast

class CodeGenerationCodeMMLUHelper(PredictionInconsistencyHumanEvalHelper, PredictionInconsistencyHelper):
    
    @staticmethod
    def _standardize_leading_whitespaces(prog: str):
        lines = prog.splitlines()
        if not lines[0].startswith('    ') and lines[0].startswith('  '):
            for idx, line in enumerate(lines):
                num_whitespaces = len(line) - len(line.lstrip())

                line = line.lstrip()
                lines[idx] = textwrap.indent(line, '  ' *num_whitespaces )
        new_prog = '\n'.join(lines)
        return new_prog
    
    @staticmethod
    def structure_mcq_choices(choices: Dict[str, str]):
        final = ""
        uppercase = string.ascii_uppercase
        idx = 0
        for choice in choices.values():
            option = uppercase[idx]
            final += f'{option}: \n{choice}\n\n'
            idx += 1
        return final
    
    @staticmethod
    def split_mutated_solution(mutated_prog: str, func_name: str) -> Tuple[str, str]:
        tree = ast.parse(mutated_prog)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                line_no = node.lineno

        lines = mutated_prog.splitlines() 
        rest_prog_lines, func_def_lines= lines[line_no:], lines[:line_no]
        return "\n".join(func_def_lines), '\n'.join(rest_prog_lines)
