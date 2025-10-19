# CodeLLM Consistency Testing

A research framework for evaluating the consistency and reliability of Large Language Models (LLMs) in code generation tasks through systematic mutation testing and behavioral analysis.

## Overview

This project investigates LLM code consistency by testing LLMs with code mutation. Code inconsistency refers to instances where a LLM could only answer 1 of 2 semantically identical problems correctly. Models with high code inconsistency scores could suggest that the models do not have a complete understanding of the program.

The research focuses on three main areas:

1. **Code Generation Testing** - Evaluating LLM performance on standard coding benchmarks
2. **Code Mutation Analysis** - Testing consistency across semantically equivalent code variations  
3. **Consistency Measurement** - Quantifying behavioral differences in LLM responses

## Key Features

### Code Mutation Types

To test the code inconsistency in LLM models, this project utilizes code mutation. In this project, code mutation refers to the process of modifying a program in at least of the following ways:

  1. syntactically (change in flow of program),
  2. lexically (changing variable names) and/or,
  3. logically (changing logical statement conditions)

while preserving the semantics of the original program. The framework supports multiple types of code mutations to test LLM consistency that fall in 1 of the 3 aforementioned types:

- **Lexical Mutations**:
    - **Random Mutation**: Mutating function names and input variable names in function definitions to random gibberish
    - **Sequential Mutation**: Mutating function names and input variable names in function definitions to generic names (E.g.: generic_function1, var1)
    - **Literal Format Mutation**: Standardising of double quotation (") or single quotation marks (') for strings in the program
- **Syntactic Mutations**:
    - **for2while**: Changing for loops to while loops in original programs
    - **for2enumerate**: Standardising for loop iterators to enumerate iterators
- **Logical Mutations**:
    - **DeMorgan Transformation**: Applying [DeMorgan](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) transformation onto boolean statements
    - **Boolean Literal**: Converting boolean literal representations. E.g.: True -> not False
    - **Comutative Reorder**: Applies semantic preserving commutative operations
    - **Constant Unfolding**: Randomly unfolds constant expressions into equivalent multiplication or addition statements
    - **Constant Unfolding Addition**: Unfolds constant expressions into addition statements
    - **Constant Unfolding Multiplication**: Unfolds constant expressions into multiplication statements
 
  _Note: The mutations will work for **most** questions, but not **all**. For example, `for2while` will fail to mutate some of the questions in the dataset. This will usually come up as a `MutationFailedError` or equivalent. Should you encounter an error like this, do ignore that question._

### Supported LLM Models

- Mistral AI models (via LangChain)
- Extensible architecture for adding new models

### Testing Frameworks

- **HumanEval Integration**: Uses the HumanEval dataset for standardized code generation benchmarks
- **Multi-process Execution**: Parallel testing for efficiency
- **MongoDB Storage**: Persistent storage for test results and analysis
- **CSV Logging**: Detailed result tracking and export capabilities

## Getting Started

### Downloading this repository
Your first step is to download this repository and save it locally on your computer. It is recommended that you use [VSCode](https://code.visualstudio.com/) for this project. VSCode is a lightweight, open-source code editor developed by Microsoft that you will use for writing code. 

### Creating your `.env` file
Your next step is to create and populate your `.env` file. A `.env` file is used to store sensitive details such as API keys and Personal Access Tokens to your personal accounts. This file will only be stored locally and should not be pushed into your GitHub.

To create your `.env` file, simply copy the `.env.example` file and rename it to `env`. For running this project locally, you will only need to fill in the "MISTRAL_API_KEY" and "MONGODB_URI" fields. If you plan to use Google Colab, you can contact the team for more information as the setup is slightly different.

### MongoDB
This project stores the dataset in MongoDB databases. Hence, you will need a MongoDB URI to store the datasets.

  1. Head to [MongoDB](https://www.mongodb.com/) and create a new account or sign in to an existing account
  2. Create a new project in MongoDB and give it a suitable name. A project houses multiple clusters, which are isolated database environments that you can use to manage, scale, and monitor your applications independently.
  3. Create a new cluster in your project. At the top of the page, you should have the option to choose between several configurations. Choose the "Free" option and choose a suitable name for your cluster. The "Free" tier will suffice for this project.
  4. Upon creating your cluster, you should be presented with a pop-up with 2 sections.
     
     The first section involves whitelisting IP address to access your cluster. You can go with "Allow Access From Anywhere" and add it to your "Network Access". When you head to the Network Access tab afterwards, you should see the IP address "0.0.0.0/0" in the list.

     The second section involves creating users for your database. Create your first user for the database. Choose an appropriate username and password for this user. Should your team wish to share a single database, you can head to the "Database Access" tab and create more users from there.
  6. Now, you will need to connect to your cluster to add and pull data from the databases in the cluster. To connect to the cluster from VSCode, navigate to the "Clusters" page and click on "Connect". Then, select the option "MongoDB for VSCode" and follow the steps on the pop-up page. From there, you will form your MongoDB Connection String (URI)

     For example, if your cluster name is "Question_Database", your username is "AlexRider" and your password is "Ark_Angel", then your MongoDB URI should look something like this: "mongodb+srv://AlexRider:Ark_Angel@Question_Database.ovenrr0.mongodb.net/". Save your MongoDB URI in your `.env` file under the name "MONGODB_URI"

     This cluster will be used to store collections, which will house the datasets that you will be using. 
  8. I would recommend that you download [MongoDB Compass](https://www.mongodb.com/products/tools/compass) for an intuitive UI to view any changes / entries in the MongoDB database conveniently. Alternatively, you can still use the MongoDB webpage to view the database.

### Mistral

You can start running some experiments with LLMs by Mistral AI. Mistral AI is a French startup (founded in 2023) that builds high-performance large language models, many of which are open-source. You will need the Mistral API to call its models. 

  1. To obtain your own Mistral API Key, simply head to the [Mistral AI](https://mistral.ai/) website and  sign up or sign in to an existing account. Complete any sign up procedures.
  2. Next, navigate to the homepage of your account. On the left hand side of the homepage, you should see a tab called "API Keys". Head to that page and create a new key. You may leave the expiration date empty.
  3. On the next pop-up, you should be presented with the API key. Copy down the API key onto your `.env` file under the name "MISTRAL_API_KEY".

By this step, you should have your "MISTRAL_API_KEY" and "MONGODB_URI" fields filled in your `.env` file.

## Running the notebooks
There are two types of notebooks: **database builder notebooks** and **notebooks for running experiments**. Database builder notebooks build your database on MongoDB using `.csv` dataset files downloaded from HuggingFace. On the other hand, experiment notebooks are used for running code inconsistency experiments.

### Creating a virtual environment
Before running the notebooks, you will need to create a Python Virtual Environment (venv). A virtual environment is an isolated workspace that allows you to install and manage project-specific dependencies without affecting your system-wide Python installation. It is good practice to always create an isolated venv for each of your Python project. 

Do note that this set up process is specific to users using the MacOS. If you are using Windows or any other OS, you may still follow these steps, but some terminal commands will not work as intended and you will need to do some troubleshooting by yourself. 

- _Skip this section if you already have Python installed and working on your computer_ -
This project uses Python version 3.11.4. You can download this version of Python for your OS from the [Python website](https://www.python.org/) or through [Anaconda](https://www.anaconda.com/). Once you have downloaded Python, ensure that you note down the file path and use the correct Python version for creating your venv. Do note that other versions of Python _could_ work as well, though the experiment was set up and run using Python 3.11.4.

Once you have Python set up, ensure that you are in the correct project folder. Then, you can create a venv using the following command in your terminal:
```
python -m venv venv
```

If you are using MacOS, activate the venv using the following command (Windows will need another command):
```
source venv/bin/activate
```

Then, install the Python packages using pip:
```
pip install -r requirements.txt
```

Wait for the packages to finish installing. You may need to do some troubleshooting should you run into any dependency installation conflicts at this stage. 

### Building the database
  1. Ensure that you have the datasets downloaded in `.csv` format. From the project's main directory, the csv datasets should be under `datasets/open_ended_format`. You should have datasets for [BigCodeBench](https://arxiv.org/abs/2406.15877), [CodeMMLU](https://arxiv.org/abs/2410.01999), [CruxEval](https://arxiv.org/abs/2401.03065), and a modified HumanEval dataset. You may refer to the links on each of the datasets to the research papers for the datasets to understand more.

     Benchmark datasets are used to grade the performance of LLMs. Each benchmark can focus on different aspects, such as general knowledge, chemistry, biology, mathematics and more. In the case of this project, the benchmark datasets we are using are focused on Python programming. 
  2. Next, from the project's root directory, navigate to the folder `1. hackathon_notebooks`. You should see 3 folders here, namely `code_generation`, `input_output_prediction` and `mcq_inconsistency`. Choose the task that you wish start with. In this tutorial, we will only be running through the process with the `HumanEval` dataset. However, other datasets should have an identical process. 
  3. Navigate to `code_generation/humaneval_database_builder.ipynb`. Select the venv that you created and run all the cells.
  4. Once all cells in the notebook have finished running successfully, you may check the dataset in your MongoDB cluster.

#### Notes for database building
  1. You will need to run `code_generation/humaneval_database_builder.ipynb` before you can run `input_output_prediction/humaneval_database_builder.ipynb` else it will fail. The input_output_prediction for humaneval database builder relies on the code generation counterpart.
  2. When building the database for BigCodeBench dataset, there may some "residual" files from the dataset that will appear in the directory. You can delete these files without any problems. 

### Running the experiments
Once you have downloaded the dataset and it's been successfully stored in your MongoDB cluster, you can start running experiments. Continuing from the steps above, we will be running through the steps for code generation tasks only. However, the steps should be identical for other experiments. Do note that you **will** need to run the corresponding dataset_builder Python notebook before you can run any experiments.

  1. Navigate to `code_generation/code_generation_experiments.ipynb`
  2. Run all the cells. For the method `run_code_generation_test`, there are some input parameters that you can modify accordingly. In short, these are some parameters that you will need:
     
     a. `prompt_helper` [Callable]: This parameter expects a `Callable` function input. This function input should return the appropriate prompt template that you wish to use.
     
     b. `num_tests` [int]: An integer indicating the number of tests you wish to run.
     
     c. `prompt_type` [str]: The prompt type, which can either be ZERO_SHOT, ONE_SHOT or FEW_SHOT. If you wish to add more prompt variations, feel free to add on to the source code.
     
     d. `output_file_path` [str]: The output file path where the results of the run will be stored in.
     
     e. `task_set` [str]: The benchmark dataset to run on.
     
     f. `continue_from_task` [str] = None: The `_id` of the task to continue running the experiment from. This comes in handy should your experiment fail.
     
     g. `mutations` [List[str]] = None: A list of strings representing the type of mutations you wish to run. 
     
  Do read the docstring of the respective methods for more information.

### Identify Code Inconsistencies
To start out with obtaining code inconsistencies in LLMs, you conduct a simple experiment.
  1. Run a code generation experiment with **no mutations** on the code generation experiment Python notebook. Your `run_code_generation_test` input should look something like the snippet below.
  ```
    pass_count = llmtester.run_code_generation_test(
      prompt_helper = OpenEndedPromptTemplate().return_appropriate_prompt(prompt_type),
      num_tests = num_tests,
      prompt_type = prompt_type,
      output_file_path = output_file_path,
      task_set = task_set,
      )
  ```
  3. Run a second experiment with **random mutation**. Your `run_code_generation_test` input should look something like the snippet below.
  ```
    pass_count = llmtester.run_code_generation_test(
      prompt_helper = OpenEndedPromptTemplate().return_appropriate_prompt(prompt_type),
      num_tests=num_tests,
      prompt_type= prompt_type,
      output_file_path=output_file_path,
      task_set = task_set,
      mutations = [RANDOM_MUTATION]    # RANDOM_MUTATION should be declared in one of the earlier cells in the same notebook
      )
  ```
  
  4. Compare the output logs and determine where the inconsistencies are. 

#### Adding new LLM Models
Should you wish to add on more models to the project, you can navigate to `llm_models/code_llms.py` and add a new class for running your desired LLM there. You will need to ensure that your new class inherits from the `CodeLLM` class, and contains the abstract methods as stated.

## Research Applications

This framework enables research into:

- **LLM Robustness**: How consistent are models across equivalent inputs?
- **Mutation Impact**: Which code transformations most affect LLM performance?
- **Model Comparison**: Systematic comparison of different LLM architectures
- **Prompt Engineering**: Effect of different prompting strategies on consistency

## Data Analysis

The framework generates detailed logs for analysis:

- Test execution results and timing
- Code mutation success/failure rates  
- LLM response variations across mutations
- Statistical consistency metrics

Analysis notebooks in `test_notebooks/` provide examples of result visualization and statistical analysis.

## Contributing

This is an active research project. Contributions are welcome in areas such as:

- Additional mutation types
- New LLM model integrations
- Enhanced analysis tools
- Performance optimizations

## Research Context

This work contributes to understanding LLM reliability in software engineering applications, with implications for:

- Automated code generation tools
- LLM-assisted programming environments  
- Robustness testing for AI-generated code
- Benchmark development for code LLMs
