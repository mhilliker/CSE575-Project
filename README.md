# CSE575-Project

Class project for statistical machine learning at Arizona State University.  Our objective is to create a predictive model to predict the response to a marketing campaign given various demographics about a set of individuals from the KDD-CUP-98 dataset.

## Setup
The environment.yml file should have all of the dependencies for this project such as sklearn, tensorflow, and keras.  To create your version of the environment using this file, install [Anaconda](https://www.anaconda.com/download/) and run the following commmand:
 ```bat
 conda env create --file environment.yml
 ```
 If this doesn't work, then you may need to install dependencies manually.
 
 Additionally, you will need to include your own copies of cup98VAL.txt and cup98LRN.txt because they are too large to fit on Github.
 
 ## Project Structure
* main.py - Main model implementations
* utils.py - Random utility functions that were created along the way
* models.ipynb - Intended for individual use for experimenting, only contains a simple example, unused in master branch
* environment.yml - YAML file that can be used by anaconda to import dependencies
