
# Analyzing Gender Representation in Multilingual Models

This project includes the experiments described in the [paper](https://arxiv.org/pdf/2204.09168.pdf): 

We have further worked on the existing project adding more languages to it and checking it for robustness.

## Prerequisites

* Python 3
* Required INLP code is already in this repository

## Data
Please download and extract relevant data files from this [link](https://drive.google.com/file/d/1j-5qdcJcqo7DHcvxuC-vLvhP1EHxGUwK/view?usp=share_link), save the data directory parallel to the src directory.
The data consists of EN, ES, FR, along with the newly added DE and IT languages in the folder. The data folder has pre trained data in inlp_matrices folder which can be used to reproduce the results quickly whihout training for each language.


## Extract representations for the datasets
For English, use src/data/run_bias_bios.sh 

For French, use src/data/run_bias_bios_fr.sh 

For Spanish, use src/data/run_bias_bios_es.sh

For German, use src/data/run_bias_bios_de.sh

For Italian, use src/data/run_bias_bios_it.sh


## Train INLP
You can use the pretrained INLP matrices under the data directory (as detailed above).
To train from scratch, use the script src/train_inlp.py. 

Example:
```
python train_inlp.py --lang EN --iters 300 --type avg --output_path <your-output-path>
```
## Explained similarity between classifiers

Use the script **src/pca_gender_repr.ipynb**.

## Gender Prediction Accuracy across Languages
Use the script **src/acc_across_langs.ipynb**

## Gender and profession classification
Use the scripts **src/classify_gender.py** and **src/classify_prof.py**
Use the script **src/classify_gender.py** Example ```python classify_gender.py --lang <select-language> ``` for gender classification on selected language
and ```python classify_prof.py --lang <select-language>``` for profession classification on the selected language

## Checking for robustness
For checking robustness, we have created a dataset containing data examples with some incorrect values. To encode the dataset, use src/data/encode_robustness_data.sh. For training use src/train_robustness_data.py, which again can be used from pretrained matrices under data/robustness/inlp_matrices. The robustness accuracy on all languages is given in src/acc_across_langs_new.ipynb





