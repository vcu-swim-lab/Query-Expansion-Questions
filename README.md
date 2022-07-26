# Paper: Using Clarification Questions to Improve Software Developers' Web Search
In the following, we briefly describe the different components that are included in this project and the softwares required to run the experiments.

## Project Structure
The project includes the following files and folders:

  - __/annotation__: A folder that contains the annotation of 200 queries
	  - annotated_queries.csv: CSV file that 200 annotated queries with facets and clarification questions
  - __/dataset__: A folder that contains inputs that are used for the experiments
	  - gen-queries.csv: CSV file that 5121 queries with regarding clarification questions
	  - template.csv: 16 clarification questions and their common answers
	  - test_dataset.csv: the 40 test queries from original 200 queries (20%)
  - __/embeddings__: A folder that contains the embeddings we have used
  - __/outputs__: A folder where outputs will be saved
  - __/models__: Contains the scripts for running the experiment
	  - model_CNN_blstm.py: the neural network model we have used for this experiment
  - __/scripts__: Contains the scripts for running the experiment
  - run_main.sh: The entry point of the experiment
  - requirements.txt: The python libraries used in this experiment
  

## Software Requirements
We have listed required software and their version to run our experiments in requirements.txt.

## Setup
1. setup virtual environment and activate it
2. `pip install -r requirements.txt'
3. `python -m spacy download en_core_web_sm'
4. Download embeddings from the following link and upzip it in the main directory: https://drive.google.com/file/d/1ONJ_OeIvjVNxJudTwq0MrDkWBslLvP7S/view?usp=sharing


## Running Experiments
Step 1: Install software requirements following the above instructions.

Step 2: Update the filepaths and parameters in *run_main.sh*

Step 3: `./run_model.sh`

## Paper link: 

https://doi.org/10.1016/j.infsof.2022.107021

## Citation:

```
@article{IMRAN2022107021,
	title = {Using clarification questions to improve software developers’ Web search},
	journal = {Information and Software Technology},
	pages = {107021},
	year = {2022},
	issn = {0950-5849},
	doi = {https://doi.org/10.1016/j.infsof.2022.107021},
	url = {https://www.sciencedirect.com/science/article/pii/S0950584922001434},
	author = {Mia Mohammad Imran and Kostadin Damevski},
}
```
