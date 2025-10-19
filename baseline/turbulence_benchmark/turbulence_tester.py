from baseline.turbulence_benchmark.utility.helper_functions import TurbulenceBenchmarkHelper
from code_generation.code_generation_tester import CodeGenerationTester, LLMExecutionRuntimeError
from typing import List, Callable, Any
from utility.constants import CodeGeneration, ReasoningModels, NonReasoningModels, LexicalMutations, SamplingMethods, Turbulence, InputPrediction, OutputPrediction, Seed
from code_mutation.mutation_relations import check_for_mutation_conflicts
from tqdm import tqdm
import ast
import random
from code_mutation.mutation_functions import CodeMutator, IdenticalMutationError
from prediction_inconsistency.prompt_templates.prompt_template import PredictionInconsistencyPromptTemplate
 
INPUT_PREDICTION = InputPrediction.NAME
OUTPUT_PREDICTION = OutputPrediction.NAME
CODE_GENERATION = CodeGeneration.NAME

class TurbulenceTester(CodeGenerationTester):
    def process_llm_predicted_output(prog: str) -> Any:
        try:
            return ast.literal_eval(prog)
        except Exception:
            return prog.strip('"').strip("'") if isinstance(prog, str) else prog
        
    def run_code_generation_test(
            self,
            output_file_path: str,
            num_tests: int,
            continue_from_task: str = None,
            mutations: List[str] = None,
            model_name: str = NonReasoningModels.GPT4O['name'],
            sampling_method: str = None,
            num_samples_per_task: int = 20,
        ):
        """
        This method is used for comparing MuCoCo code generation testing with the Turbulence benchmark.

        Since this is a baseline testing class and it is not the primary focus of the project, this method does not support GPU functionalities like other tester methods and can only run experiments through a local setup with API calls.
        """

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        timeout = 8                     # int variable indicating the number of seconds the LLM generated program should complete running by
        
        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('Q')[-1])
        else:
            continue_from = 1

        # ensuring that number of samples per task is <= max_samples_per_task, which is the maximum possible number of samples per task
        max_samples_per_task = len(self.question_database.find_one({"_id" : "TurbulenceQ1"})['params'])
        num_samples_per_task = min(num_samples_per_task, max_samples_per_task)

        num_tests = min(self.question_database.count_documents({}) - continue_from + 1, num_tests + 1)         # ensuring that the number of iterations is lower than max number of documents in the db
        if not check_for_mutation_conflicts(mutations=mutations):
            raise ValueError("An invalid combination of mutations were used.")

        for mutation in mutations:
            if mutation not in CodeGeneration.MUTATIONS or mutation == LexicalMutations.LITERAL_FORMAT:
                raise ValueError(f"{mutation} mutation is an invalid mutation for code generation.")

        reasoning_models = [getattr(ReasoningModels, model) for model in dir(ReasoningModels) if not model.startswith("_")]
        non_reasoning_models = [getattr(NonReasoningModels, model) for model in dir(NonReasoningModels) if not model.startswith("_")]
        all_local_models = reasoning_models + non_reasoning_models

        for model in all_local_models:
            if model['name'] == model_name:
                llm = model['model_class']
                break
        else:
            valid_model_names = [model['name'] for model in all_local_models]
            raise ValueError(f"{model_name} is not a valid local model. The models supported by this framework are {', '.join(valid_model_names)}")

        try:     # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
            for q_no in tqdm(range(continue_from, continue_from + num_tests+1), desc = "Question Progress"):
                task_id = f"TurbulenceQ{q_no}"

                qn_sample = self.question_database.find_one({"_id": task_id})

                if qn_sample is None:                                           # skip to the next task if unable to extract the specific qn id from MongoDB
                    print(f"Could not retrieve {task_id} from the MongoDB Turbulence Dataset")
                    continue
            
                tests_template: str = qn_sample['question_template']              # contains the question without the docstring description
                prompt_template: str = qn_sample['prompt_template']               # doc string description
                solution_template: str = qn_sample['solution_template']           # dict object containing all examples pertaining the question for few shot/one shot prompting
                params_dict: dict = qn_sample['params']                           # canonical solution to the question
                func_name: str = qn_sample['func_name']                           # check function for testing validity of a solution

                # Sampling of tasks. There are a total of 100 variations of each tasks, which can be both cost and computationally expensive, hence conducting the experiments on a sample size is more ideal.
                if sampling_method == SamplingMethods.RANDOM:
                    test_keys = random.choices(params_dict.keys() , k = num_samples_per_task)
                    test_params = [params_dict[test_key] for test_key in test_keys]
                elif sampling_method == SamplingMethods.SYSTEMATIC:
                    step = max_samples_per_task // num_samples_per_task
                    count = 0
                    test_params = []
                    while count < max_samples_per_task and len(test_params) < num_samples_per_task:
                        test_params.append(params_dict[str(count)])
                        count += step
                else:
                    test_params = list(params_dict.values())   


                for idx, task in tqdm(enumerate(test_params), desc=f"Neighbourhood Task", position = 1):
                    
                    # Dictionary storing all relevant log data
                    log_data_entry = {
                        "task_id": f"{task_id}_{idx+1}",
                        "prompt" : None,
                        "model_output": None,
                        "check_function": None,
                        "canonical_solution": None,
                        "func_input": None,
                        "func_output": None,
                        "failure_type": None,
                    }

                    params = task['params']
                    func_input = task['func_input']
                    func_output = task['func_output']

                    func_name: str = qn_sample['func_name']                           # check function for testing validity of a solution
                    
                    helper = TurbulenceBenchmarkHelper()

                    solution = solution_template
                    tests = tests_template
                    prompt = prompt_template

                    processed_params = helper.convert_data_to_metadata(data = params["data"], metadata = params['metadata'])
                    processed_func_input = helper.convert_data_to_metadata(data = func_input["data"], metadata = func_input['metadata'])
                    processed_func_output = helper.convert_data_to_metadata(data = func_output["data"], metadata = func_output['metadata'])

                    log_data_entry['func_input'] = processed_func_input

                    for param_idx, processed_param in enumerate(processed_params):
                        solution = solution.replace(f"${param_idx}", str(processed_param))
                        tests = tests.replace(f"${param_idx}", str(processed_param))
                        prompt = prompt.replace(f"${param_idx}", str(processed_param))
                    
                    tests = helper.replace_func_name(tests_template = tests, func_name = func_name)

                    try:
                        helper.run_test_suite(tests = tests, solution = solution)
                    except Exception as e:
                        print(f'{task_id} failed the tests when retrieved')
                        log_data_entry['failure_type'] = MemoryError()
                        TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                        continue

                    # Handling Task Mutation (If any)
                    codemutator = CodeMutator(
                        func_name=func_name,
                        mutated_dict = {
                            'question': solution,
                            'qn_desc': prompt,
                            'check_function' : tests,
                            'full_sol' : solution
                        },
                        benchmark_set=Turbulence.NAME
                    )

                    for mutation in mutations:
                        codemutator.handle_mutation(
                            mutation_type=mutation,
                            task_set=Turbulence.NAME,
                            tree = ast.parse(codemutator.mutated_dict['question'])
                        )
                        func_name = codemutator.func_name

                    log_data_entry['canonical_solution'] = codemutator.mutated_dict['full_sol']
                    log_data_entry["prompt"] = codemutator.mutated_dict['qn_desc']
                    log_data_entry['check_function'] = codemutator.mutated_dict['check_function']

                    input_variables = {"prompt" : log_data_entry["prompt"] }
                    
                    # Running the llm on the input variables and the prompt template
                    try: 
                        ans = self.execute_llm(input_variables = input_variables, prompt_template = "{prompt}", llm_model = llm, model_name=model_name)
                    
                    except Exception as e:
                        log_data_entry["failure_type"] = (LLMExecutionRuntimeError.__name__, type(e))
                        TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                        continue
                    try: 
                        # Processing of the llm answer. Some llm answers are in Python code blocks, which needs to be processed as it will fail exec()
                        ans = TurbulenceTester.process_llm_ans(ans)
                    
                    except ValueError:                          # Raised when the llm answer did not have a python code block
                        try: 
                            exec(ans)                           # Attempting to run the llm answer directly. In some cases, the returned answer can be directly run as no code block was returned
                        except Exception as e:                  # Else, if the answer is not in a valid code block and cannot be run directly, it is a faulty answer and is stored accordingly.
                            print(f"Could not process LLM answer: {e}")
                            log_data_entry["model_output"] = ans
                            log_data_entry["failure_type"] = ("could_not_parse_LLM_answer", type(e))
                            TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                            continue


                    log_data_entry['model_output'] = ans       # storing the answer in input_data dict
                    ## LLM Answer Test Execution    
            
                    try:
                        # Verification Step 1: Verifying LLM answer with canonical solution
                        helper.verify_prog_answer(
                            canonical_sol = ans,
                            func_input = processed_func_input,
                            func_name = func_name,
                            func_output = processed_func_output
                            )
                                                
                        log_data_entry['func_output'] = processed_func_output

                        # Verification Step 2: Verifying LLM answer with test suite
                        helper.run_test_suite(
                            tests = codemutator.mutated_dict['check_function'],
                            solution = ans,
                        )
                        
                        task_pass_count += 1

                    except Exception as e:
                        if isinstance(e, AssertionError):
                            print("{task_id}: Function failed to run due to following error -> {e}".format(e = e, task_id = task_id))
                        elif isinstance(e, RuntimeError):
                            print("{task_id}: LLM Answer exceeded runtime of {timeout} seconds -> {e}".format(e = e, task_id = task_id, timeout = timeout))
                        else:
                            print("{task_id}: Could not run the LLM answer due to the following error: {e}".format(e = e, task_id = task_id))
                        log_data_entry['failure_type'] = f"{type(e)} > {e}"            # Logging failure type into log_data_entry

                    # logging completed run into csv 
                    TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                
            return task_pass_count
        except Exception as e:
            print(e)
            print(task_id)
            return task_pass_count
        
        except KeyboardInterrupt:
            print(task_id)
            return task_pass_count
        

    
    def run_prediction_inconsistency_test(
            self,
            output_file_path: str,
            num_tests: int,
            task_type: str,
            continue_from_task: str = None,
            mutations: List[str] = None,
            model_name: str = NonReasoningModels.GPT4O['name'],
            sampling_method: str = None,
            num_samples_per_task: int = 20,
        ):
        """
        This method is used for comparing MuCoCo code generation testing with the Turbulence benchmark.

        Since this is a baseline testing class and it is not the primary focus of the project, this method does not support GPU functionalities like other tester methods and can only run experiments through a local setup with API calls.
        """
        if task_type == INPUT_PREDICTION:
            prediction_prompt_template = PredictionInconsistencyPromptTemplate.InputPrediction().zero_shot_prompt()
        elif task_type == OUTPUT_PREDICTION:
            prediction_prompt_template = PredictionInconsistencyPromptTemplate.OutputPrediction().zero_shot_prompt()
        else:
            raise ValueError(f"Invalid task_type was used. The only valid task types are{INPUT_PREDICTION, OUTPUT_PREDICTION}")

        task_pass_count = 0             # int variable tracking the number of tasks that have passed
        
        if continue_from_task is not None:
            continue_from = int(continue_from_task.split('Q')[-1])
        else:
            continue_from = 1

        # ensuring that number of samples per task is <= max_samples_per_task, which is the maximum possible number of samples per task
        max_samples_per_task = len(self.question_database.find_one({"_id" : "TurbulenceQ1"})['params'])
        num_samples_per_task = min(num_samples_per_task, max_samples_per_task)

        total_tests = self.question_database.count_documents({}) + 1 
        num_tests = min(total_tests - continue_from, num_tests)         # ensuring that the number of iterations is lower than max number of documents in the db
        if not check_for_mutation_conflicts(mutations=mutations):
            raise ValueError("An invalid combination of mutations were used.")

        for mutation in mutations:
            if mutation not in OutputPrediction.MUTATIONS:
                raise ValueError(f"{mutation} mutation is an invalid mutation for code generation.")

        reasoning_models = [getattr(ReasoningModels, model) for model in dir(ReasoningModels) if not model.startswith("_")]
        non_reasoning_models = [getattr(NonReasoningModels, model) for model in dir(NonReasoningModels) if not model.startswith("_")]
        all_local_models = reasoning_models + non_reasoning_models

        for model in all_local_models:
            if model['name'] == model_name:
                llm = model['model_class']
                break
        else:
            valid_model_names = [model['name'] for model in all_local_models]
            raise ValueError(f"{model_name} is not a valid local model. The models supported by this framework are {', '.join(valid_model_names)}")

        # try:     # try statement to catch any potential errors arising from using free APIs. These APIs are usually unstable and can crash at any time. 
        for q_no in tqdm(range(continue_from, continue_from + num_tests + 1), desc = "Question Progress"):
            task_id = f"TurbulenceQ{q_no}"

            qn_sample = self.question_database.find_one({"_id": task_id})

            if qn_sample is None:                                           # skip to the next task if unable to extract the specific qn id from MongoDB
                print(f"Could not retrieve {task_id} from the MongoDB Turbulence Dataset")
                continue
        
            tests_template: str = qn_sample['question_template']              # contains the question without the docstring description
            prompt_template: str = qn_sample['prompt_template']               # doc string description
            solution_template: str = qn_sample['solution_template']           # dict object containing all examples pertaining the question for few shot/one shot prompting
            params_dict: dict = qn_sample['params']                           # canonical solution to the question

            # Sampling of tasks. There are a total of 100 variations of each tasks, which can be both cost and computationally expensive, hence conducting the experiments on a sample size is more ideal.
            if sampling_method == SamplingMethods.RANDOM:
                random.seed(Seed.value)
                test_keys = random.choices(list(params_dict.keys()) , k = num_samples_per_task)
                test_params = [params_dict[test_key] for test_key in test_keys]
            elif sampling_method == SamplingMethods.SYSTEMATIC:
                step = max_samples_per_task // num_samples_per_task
                count = 0
                test_params = []
                while count < max_samples_per_task and len(test_params) < num_samples_per_task:
                    test_params.append(params_dict[str(count)])
                    count += step
            else:
                test_params = list(params_dict.values())   


            for idx, task in tqdm(enumerate(test_params), desc=f"Neighbourhood Task", position = 1, leave = False):
                
                # Dictionary storing all relevant log data
                log_data_entry = {
                    "task_id": f"{task_id}_{idx+1}",
                    "prompt" : None,
                    "model_output": None,
                    "check_function": None,
                    "canonical_solution": None,
                    "func_input": None,
                    "func_output": None,
                    "failure_type": None,
                }

                params = task['params']
                func_input = task['func_input']
                func_output = task['func_output']

                # func_name has to be reset each iteration as lexical mutaiton may mutate it.
                func_name: str = qn_sample['func_name']                           # check function for testing validity of a solution
                
                helper = TurbulenceBenchmarkHelper()

                solution = solution_template
                tests = tests_template
                prompt = prompt_template

                processed_params = helper.convert_data_to_metadata(data = params["data"], metadata = params['metadata'])
                processed_func_input = helper.convert_data_to_metadata(data = func_input["data"], metadata = func_input['metadata'])
                processed_func_output = helper.convert_data_to_metadata(data = func_output["data"], metadata = func_output['metadata'])

                log_data_entry['func_input'] = processed_func_input
                log_data_entry['func_output'] = processed_func_output

                ## For loop iterating through each parameter and replacing the corresponding points in the solution, test and prompt template
                for param_idx, processed_param in enumerate(processed_params):
                    solution = solution.replace(f"${param_idx}", str(processed_param))
                    tests = tests.replace(f"${param_idx}", str(processed_param))
                    prompt = prompt.replace(f"${param_idx}", str(processed_param))
                
                ## Replacing the function name in templates. This is done as a seperate function as there are unintended replacements when using .replace(), hence a regex pattern is more stable
                tests = helper.replace_func_name(tests_template = tests, func_name = func_name)
                try:
                    ## Verifying that the canon solution passes the test suite
                    helper.run_test_suite(tests = tests, solution = solution)
                    
                    ## Verifying that the canonical solution returns the same output as the output stored in MongoDB
                    helper.verify_prog_answer(canonical_sol=solution, func_input=processed_func_input, func_name = func_name, func_output=processed_func_output)
                
                except Exception as e:
                    print(f'{task_id} failed the tests when retrieved')
                    log_data_entry['failure_type'] = MemoryError()
                    TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                    continue

                # Handling Task Mutation (If any)
                codemutator = CodeMutator(
                    func_name=func_name,
                    mutated_dict = {
                        'question': solution,
                        'qn_desc': prompt,
                        'check_function' : tests,
                        'full_sol' : solution
                    },
                    benchmark_set=Turbulence.NAME
                )
                try:
                    for mutation in mutations:
                        codemutator.mutate_for_prediction_inconsistency_test(
                            mutation_type=mutation,
                            task_set=Turbulence.NAME,
                            input_args=processed_func_input,
                            output_args=processed_func_output,
                            input_metadata=func_input['metadata']
                        )
                        func_name = codemutator.func_name
                except Exception as e:
                    log_data_entry["failure_type"] = f"{type(e)} > {e}"
                    TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                    continue


                modified_qn_desc = helper.modify_original_prompt_for_prediction_testing(codemutator.mutated_dict['qn_desc'])

                input_variables = {
                    "qn_desc" : modified_qn_desc,
                    "full_sol" : codemutator.mutated_dict['full_sol'],
                    "test_output" : processed_func_output,
                    "test_input" : processed_func_input,
                    }

                log_data_entry['canonical_solution'] = codemutator.mutated_dict['full_sol']
                log_data_entry["prompt"] = prediction_prompt_template.format(**input_variables)
                log_data_entry['check_function'] = codemutator.mutated_dict['check_function']

                # Running the llm on the input variables and the prompt template
                try: 
                    ans = self.execute_llm(
                        input_variables = input_variables, 
                        prompt_template = prediction_prompt_template, 
                        llm_model = llm, 
                        model_name=model_name
                    )
                except Exception as e:
                    log_data_entry["failure_type"] = f"{type(e)} > {e}"
                    TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                    continue

                try: 
                    # Processing of the llm answer. Some llm answers are in Python code blocks, which needs to be processed as it will fail exec()
                    ans = TurbulenceTester.process_llm_predicted_output(ans)
                
                except Exception as e:                  # Else, if the answer is not in a valid code block and cannot be run directly, it is a faulty answer and is stored accordingly.
                    print(f"Could not process LLM answer: {e}")
                    log_data_entry["model_output"] = ans
                    log_data_entry["failure_type"] = ("could_not_parse_LLM_answer", type(e))
                    TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
                    continue

                log_data_entry['model_output'] = ans       # storing the answer in input_data dict

                ## LLM Answer Test Execution    
                try:
                    if task_type == OUTPUT_PREDICTION:
                        # Verification Step: Verifying LLM Output with canonical solution output
                        helper.verify_prog_output(
                            canon_ans=processed_func_output,
                            func_output=ans
                        )
                    else:
                        assert ans == True
                    
                    task_pass_count += 1

                except Exception as e:
                    # if isinstance(e, AssertionError):
                    #     print("{task_id}: Function failed to run due to following error -> {e}".format(e = e, task_id = task_id))
                    # elif isinstance(e, RuntimeError):
                    #     print("{task_id}: LLM Answer exceeded runtime of {timeout} seconds -> {e}".format(e = e, task_id = task_id, timeout = timeout))
                    # else:
                    #     print("{task_id}: Could not run the LLM answer due to the following error: {e}".format(e = e, task_id = task_id))
                    log_data_entry['failure_type'] = f"{type(e)} > {e}"            # Logging failure type into log_data_entry

                # logging completed run into csv 
                TurbulenceTester.log_into_csv(output_file_path = output_file_path, input_data = log_data_entry)
            
        return task_pass_count
        # except Exception as e:
        #     print(mutation)
        #     print(e)
        #     print(task_id)
        #     return task_pass_count
        
        # except KeyboardInterrupt:
        #     print(task_id)
        #     return task_pass_count