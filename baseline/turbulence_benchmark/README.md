# Turbulence Benchmark Experiment Setup

This folder contains files required for running the Turbulence Benchmark.

This project sub-directory is as follows:

- `project-root/baseline/turbulence_benchmark`
  - `README.md`                       # this file
  - `Source_Code (from Turbulence GitHub)`       # Folder contains source code directly pulled from the Turbulence benchmark GitHub
  - `utility`
    - `helper_functions.py`
    - `turbulence_log_functions.py`
  - `turbulence_benchmark_test_notebook.ipynb`
  - `turbulence_database_builder.ipynb`
  - `turbulence_tester.py`

## Running Turbulence Experiments

To get started with running MuCoCo experiments against the Turbulence benchmark, follow these steps:

1. Ensure that you have a working GPT4O model and have filled in the API key in the .env file according to the .env.example.
2. Ensure that you have MongoDB setup and you have the necessary MongoDB details stored in the .env file according to the .env.example.
3. Run all cells in `turbulence_database_builder.ipynb` to build the Turbulence benchmark and store it in MongoDB. This notebook utilizes the source code from the original Turbulence benchmark to generate the Turbulence code, hence they have been included in this repository.
4. Run `turbulence_benchmark_test_notebook.ipynb` to start running the Turbulence experiments. Some sample code has been included in this notebook to get you started.
