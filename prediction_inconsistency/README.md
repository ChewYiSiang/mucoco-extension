# MuCoCo Prediction Inconsistency Set up

This folder contains files required for running MuCoCo Prediction Inconsistency Experiments.

This project sub-directory is as follows:

```markdown
project-root/prediction_inconsistency
├── README.md # this file
├── ❌ prompt_templates # prompt templates
│ ├── deepseek_prompt_template.py
│ ├── gemma_prompt_template.py
│ ├── mistral_prompt_template.py
│ ├── prompt_template.py
│ └── qwen_prompt_template.py
│
├── test_notebooks # Notebooks for pred inconsistency experiments
│ ├── crux_eval_database_builder.ipynb
│ ├── human_eval_database_builder.ipynb
│ ├── input_prediction_consistency_test_notebook_cruxeval.ipynb
│ ├── input_prediction_consistency_test_notebook_humaneval.ipynb
│ ├── output_prediction_consistency_test_notebook_cruxeval.ipynb
│ └── output_prediction_consistency_test_notebook_humaneval.ipynb
│
├── ❌ utility
│ ├── cruxeval_helper.py
│ ├── database_helper.py
│ └── humaneval_helper.py
│
└── ❌ prediction_inconsistency_tester.py
```

Files and folders marked with a '❌' do not need to be visited for running the relevant MuCoCo experiments.

## General LLM and MongoDB setup for MuCoCo Prediction Inconsistency Experiments

To get started with running MuCoCo Prediction Inconsistency experiments, follow these steps:

1. Ensure that you have at least one working LLM and have filled in the API key in the .env file according to .env.example.
2. Ensure that you have MongoDB setup and you have the necessary MongoDB details stored in the .env file according to .env.example.


## CruxEval Benchmark Setup
One of the two benchmarks used for Prediction Inconsistency task is CruxEval. Before running experiments on CruxEval, we will need to preprocess the data and store it in MongoDB.

1. Start by running all cells in `test_notebooks/crux_eval_database_builder.ipynb` to build the CruxEval benchmark and store it in MongoDB. This notebook utilizes the csv format of CruxEval downloaded from HuggingFace.
2. Check that CruxEval tasks have been successfully stored in MongoDB.
3. Run `input_prediction_consistency_test_notebook_cruxeval.ipynb` or `output_prediction_consistency_test_notebook_cruxeval,ipynb` to start running the CruxEval experiments. Some sample code has been included in this notebook to get you started.

## HumanEval Benchmark Setup

The other benchmark used for the Prediction Inconsistency task is HumanEval. Before running experiments on HumanEval, we need to preprocess the data and store it in MongoDB.

1. Start by running all cells in `test_notebooks/human_eval_database_builder.ipynb` to build the HumanEval benchmark and store it in MongoDB. This notebook utilizes the JSON format of HumanEval downloaded from HuggingFace.
2. Check that HumanEval tasks have been successfully stored in MongoDB.
3. Run `input_prediction_consistency_test_notebook_humaneval.ipynb` or `output_prediction_consistency_test_notebook_humaneval.ipynb` to start running the HumanEval experiments. Some sample code has been included in this notebook to get you started.
