# CodeLLM Consistency Testing

A research framework for evaluating the consistency and reliability of Large Language Models (LLMs) in code generation tasks through systematic mutation testing and behavioral analysis.

## Overview

This project investigates LLM consistency by testing how well code generation models handle semantically equivalent but syntactically different versions of programming problems. The research focuses on three main areas:

1. **Code Generation Testing** - Evaluating LLM performance on standard coding benchmarks
2. **Code Mutation Analysis** - Testing consistency across semantically equivalent code variations
3. **Consistency Measurement** - Quantifying behavioral differences in LLM responses

## Project Structure

```
├── code_generation/          # Code generation testing framework
│   ├── code_generation_tester.py    # Main testing class for code generation
│   ├── prompt_templates/             # Prompt engineering templates
│   ├── test_notebooks/              # Jupyter notebooks for experiments
│   └── utility/                     # Helper functions for HumanEval dataset
├── prediction_inconsistency/       # Consistency testing framework
│   ├── prediction_inconsistency_tester.py # Main testing class for consistency
│   ├── prompt_templates/             # Prompt templates for consistency tests
│   ├── test_notebooks/              # Analysis notebooks
│   └── utility/                     # HumanEval helper functions
├── code_mutation/            # Code mutation utilities
│   ├── ast_mutation.py              # AST-based code transformations
│   ├── mutation_functions.py        # Core mutation algorithms
│   └── simple_demorgan_test.py      # De Morgan's law transformations
├── llm_models/               # LLM integration layer
│   ├── code_llms.py                 # Base classes for code LLMs
│   └── code_reasoning_llms.py       # Reasoning-focused LLM variants
├── utility/                  # Shared utilities
│   ├── data_log_functions.py        # Data logging and CSV management
│   └── humaneval_dataset_download.py # HumanEval dataset utilities
├── database.py               # MongoDB integration for data storage
└── README.md                 # This file
```

## Key Features

### Code Mutation Types

The framework supports multiple types of code mutations to test LLM consistency:

- **Lexical Mutations**: Variable name changes, formatting variations
- **Syntactic Mutations**:
  - For-loop to while-loop transformations
  - For-loop to enumerate transformations
  - Condition augmentation with De Morgan's laws
- **Semantic Preserving**: All mutations maintain program semantics

### Supported LLM Models

- Meta LLaMA models (via Hugging Face)
- Mistral AI models (via LangChain)
- Extensible architecture for adding new models

### Testing Frameworks

- **HumanEval Integration**: Uses the HumanEval dataset for standardized code generation benchmarks
- **Multi-process Execution**: Parallel testing for efficiency
- **MongoDB Storage**: Persistent storage for test results and analysis
- **CSV Logging**: Detailed result tracking and export capabilities

## Getting Started

### Prerequisites

```bash
pip install pymongo langchain langchain-mistralai huggingface_hub pandas tqdm python-dotenv
```

### Environment Setup

1. Create a `.env` file with your API keys and database URI:

```bash
MONGODB_URI=your_mongodb_connection_string
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACE_API_TOKEN=your_huggingface_token
```

2. Download the HumanEval dataset:

```python
from utility.humaneval_dataset_download import download_humaneval
download_humaneval()
```

### Basic Usage

#### Code Generation Testing

```python
from code_generation.code_generation_tester import CodeGenerationTester
from llm_models.code_llms import MetaLlama

# Initialize tester and model
tester = CodeGenerationTester()
llm = MetaLlama("meta-llama/Llama-3.2-3B-Instruct")

# Run tests
tester.run_code_generation_test(
    llm=llm,
    prompt_helper=your_prompt_function,
    num_tests=100,
    output_file_path="results.csv",
    prompt_type="zero_shot"
)
```

#### Consistency Testing

```python
from prediction_inconsistency.prediction_inconsistency_tester import LLMConsistencyTester

# Initialize consistency tester
consistency_tester = LLMConsistencyTester()

# Run consistency analysis with mutations
consistency_tester.run_code_consistency_test(
    llm=llm,
    prompt_helper=your_prompt_function,
    num_tests=50,
    output_file_path="consistency_results.csv",
    prompt_type="zero_shot",
    lexical_mutation="variable_renaming",
    syntactic_mutation="for2while"
)
```

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
