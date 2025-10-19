# MuCoCo Code Generation Set up

This folder contains files required for running MuCoCo Code Generation Experiments.

This project sub-directory is as follows:

```markdown
project-root/code_generation
├── README.md # this file
├── ❌ prompt_templates # prompt templates
│ ├── deepseek_prompt_template.py
│ ├── gemma_prompt_template.py
│ ├── mistral_prompt_template.py
│ ├── prompt_template.py
│ └── qwen_prompt_template.py
│
├── test_notebooks # notebooks for code generation experiments
│ ├── bigcodebench_database_builder.ipynb
│ ├── code_generation_test_notebook_bigcodebench.ipynb
│ ├── code_generation_test_notebook_humaneval.ipynb
│ └── humaneval_database_builder.ipynb
│
├── ❌ utility 
│ ├── bigcodebench_helper.py
│ ├── database_helper.py
│ └── humaneval_helper.py
│
└── ❌ code_generation_tester.py
```

Files and folders marked with a '❌' do not need to be visited for running the relevant MuCoCo experiments.

## General LLM and MongoDB setup for MuCoCo Code Generation Experiments

To get started with running MuCoCo code generation experiments, follow these steps:

1. Ensure that you have at least one working LLM and have filled in the API key in the .env file according to .env.example.
2. Ensure that you have MongoDB setup and you have the necessary MongoDB details stored in the .env file according to .env.example.


## BigCodeBench Setup
One of the benchmarks used for code generation task is BigCodeBench. Before running experiments on BigCodeBench, we will need to preprocess the data and store it in MongoDB.

1. Start by all cells in `test_notebooks/bigcodebench_database_builder.ipynb` to build the BigCodeBench benchmark and store it in MongoDB. This notebook utilizes the csv format of BigCodeBench downloaded from HuggingFace. Do note that some of the tasks in BigCodeBench *will* fail. The tasks that will fail are recorded down in the `bigcodebench_database_builder.ipynb`, but it may change over time depending on the time of running these experiments due to issues such as package conflicts.
2. Check that BigCodeBench tasks have been successfully stored in MongoDB.
3. Run `code_generation_test_notebook_bigcodebench.ipynb` to start running the BigCodeBench experiments. Some sample code has been included in this notebook to get you started.


## HumanEval Setup
The other benchmark used for code generation task is HumanEval. Before running experiments on HumanEval, we will need to preprocess the data and store it in MongoDB.

1. Start by all cells in `test_notebooks/humaneval_database_builder.ipynb` to build the HumanEval benchmark and store it in MongoDB. This notebook utilizes the csv format of HumanEval downloaded from HuggingFace. Some minor edits have been made to the original csv for data processing and the modifications have been reported in `humaneval_database_builder.ipynb`.
2. Check that HumanEval tasks have been successfully stored in MongoDB.
3. Run `code_generation_test_notebook_humaneval.ipynb` to start running the HumanEval experiments. Some sample code has been included in this notebook to get you started.