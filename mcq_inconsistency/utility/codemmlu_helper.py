from code_inconsistency.utility.database_helper import CodeInconsistencyHelper
from code_inconsistency.utility.humaneval_helper import CodeInconsistencyHumanEvalHelper
import textwrap
from typing import List
import string

class CodeGenerationCodeMMLUHelper(CodeInconsistencyHumanEvalHelper, CodeInconsistencyHelper):
    
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
    def structure_mcq_choices(choices: List[str]):
        final = ""
        uppercase = string.ascii_uppercase
        for idx, choice in enumerate(choices):
            option = uppercase[idx]
            final += f'{option}: \n{choice}\n\n'
        return final



