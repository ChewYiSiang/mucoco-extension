from code_inconsistency.utility.database_helper import CodeInconsistencyHelper
import textwrap
import re

class CodeInconsistencyCodeMMLUHelper(CodeInconsistencyHelper):
    
    @staticmethod
    def _standardize_leading_whitespaces(prog: str):
        lines = prog.splitlines()
        if not lines[0].startswith('    ') and lines[0].startswith('  '):
            for idx, line in enumerate(lines):
                num_whitespaces = len(line) - len(line.lstrip())
                print(num_whitespaces )
                line = line.lstrip()
                lines[idx] = textwrap.indent(line, '  ' *num_whitespaces )
        new_prog = '\n'.join(lines)
        return new_prog
    



