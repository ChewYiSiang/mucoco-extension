import inspect
from typing import List

class PredictionInconsistencyHelper:
    @staticmethod
    def check_input_output(
        full_sol: str, 
        test_input: str, 
        expected_output: str, 
        func_name: str, 
        input_metadata: List[str]
    ) -> bool:
        
        namespace = {}
        try:
            exec(full_sol, namespace)
            sig = inspect.signature(namespace[func_name])
            if test_input is None:
                assert namespace[func_name]() == expected_output
            elif not isinstance(test_input, (int)) and len(sig.parameters) > 1:
                # print(namespace[func_name](*test_input), type(namespace[func_name](*test_input)))
                # print(expected_output, type(expected_output))
                assert namespace[func_name](*test_input) == expected_output
            else:
                # print(expected_output, type(expected_output))
                # print(namespace[func_name](test_input), type(namespace[func_name](test_input)))
                assert namespace[func_name](test_input) == expected_output
            return True
        except AssertionError as e:
            return False
        except Exception as e:
            print(f"Could not evaluate TF due to the following error: {e}")
            return False
    