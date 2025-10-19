# MuCoCo MCQ Inconsistency Set up

This folder contains files required for running MuCoCo MCQ Inconsistency Experiments.

This project sub-directory is as follows:

```markdown
project-root/mcq_inconsistency
├── README.md # this file
├── ❌ prompt_templates # prompt templates
│ ├── deepseek_prompt_template.py
│ ├── gemma_prompt_template.py
│ ├── llama_prompt_template.py
│ ├── mistral_prompt_template.py
│ ├── prompt_template.py
│ └── qwen_prompt_template.py
│
├── test_notebooks # Notebooks for MCQ inconsistency experiments
│ ├── codemmlu_database_builder.ipynb
│ └── mcq_inconsistency_test_notebook.ipynb
│
├── ❌ utility
│ └── codemmlu_helper.py
│
└── ❌ mcq_inconsistency_tester.py
```

Files and folders marked with a '❌' do not need to be visited for running the relevant MuCoCo experiments.


## General LLM and MongoDB setup for MuCoCo MCQ Inconsistency Experiments

To get started with running MuCoCo MCQ Inconsistency experiments, follow these steps:

1. Ensure that you have at least one working LLM and have filled in the API key in the .env file according to .env.example.
2. Ensure that you have MongoDB setup and you have the necessary MongoDB details stored in the .env file according to .env.example.


## CodeMMLU Setup
The benchmark used for MCQ Inconsistency task is CodeMMLU. Before running experiments on CodeMMLU, we will need to preprocess the data and store it in MongoDB.

1. Start by running all cells in `test_notebooks/codemmlu_database_builder.ipynb` to build the CodeMMLU benchmark and store it in MongoDB. This notebook utilizes the csv format of CodeMMLU downloaded from HuggingFace.
2. Check that CodeMMLU tasks have been successfully stored in MongoDB.
3. Run `mcq_inconsistency_test_notebook.ipynb` to start running the CodeMMLU experiments. Some sample code has been included in this notebook to get you started.