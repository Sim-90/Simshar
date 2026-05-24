README: 

"From Redundancy to Efficiency: A Similarity-based Filter Sharing for Deep Model Compression"

Prerequisites: 
1. The libraries present in the reqt_cka.txt
2. python3.8

Environment:
The virtual environment file has been given as "reqt_cka.txt" for the libraries required for the implementation. 

Directories
The main directory has 4 sub-directories:
	1. models: contains the pretrained version of the models chosen for this experiment. The models are available online. They were finetuned for the particular dataset for the 		experiments.
	2. cka_models: will contain the compressed version of the models.
	3. util: contains .py files with basic functions required for the execution.
	4. data: contain the datasets.

Files:
There are two main execution files in the main directory: "model_cka_whole.py" and "model_cka_subsets.py". 
1. "model_cka_whole.py": Contains the code for the framework application over the chosen model and data (options for model selection given in the code). This code performs the similarity-based layerwise filter shairng for the entire dataset chosen.

2."model_cka_subsets.py": This contains the framework application for the subsets generated for the chosen dataset and the model. The execution is done on the model training over the subsets of the dataset.
