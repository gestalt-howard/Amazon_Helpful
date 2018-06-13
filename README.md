# Amazon Review Helpfulness Prediction

***Project Completion Date:*** June 10, 2018

![Project Photo](./images/advertising-amazon-amazon-seo-907607.jpg)

## Project Overview:
This repository contains snippets of development code along with the final code used to deliver results for the Kaggle competition (https://www.kaggle.com/c/dse-220-2018-final).

The overarching goal of this project was to accurately predict the helpfulness of particular reviews for various Amazon products. Accuracy was evaluated using the metric of mean absolute error (MAE). The lower the MAE, the more accurate the overall predictions were.

My final standing on the private leaderboard is **9th place out of 28 with a MAE score of 0.16703**.

If you'd like to run the code that created my result in this Kaggle competition, please run ***ONLY*** the Jupyter Notebook scripts in this specified order:
1. Preprocessing.ipynb
2. MClassification_Regression.ipynb

All other Python files are vestiges of the development process which is exhaustively detailed in the ***project_summary.pdf*** file. These development scripts can be referenced while reading through the overall project report.

## File Description:
### Final Code (100% Operational):
* **Preprocessing.ipynb**: Creates preprocessed data to be used in the *MClassification_Regression.ipynb* script
* **MClassification_Regression.ipynb**: My best machine learning model consisting of a cascaded Gradient Boosting Classifier and 2 Gradient Boosting Regressors

### Project Development and Results Report:
* **project_summary.pdf**: Walks through the entire project development process and development of the final successful machine learning model

### Setup Files:
The files listed below were the original files given in the Kaggle competition:
* **test_Helpful.json.gz**: Zipped test dataframe
* **train.json.gz**: Zipped train dataframe
* **pairs_Helpful.txt**: Contains formatting to be used for making submissions on Kaggle

### Development Scripts:
* ***modules*** folder: Contains helper functions used for the development scripts
* **corpus_gen.py**: Formats the text corpus from the original zipped train and test dataframes for usage in text preprocessing
* **data_gen.py**: Generated preprocessed data used in development phase
* **debug_gen.py**: Genearted reduced dataset used in development
* **reduced_gen.py**: A vestige of the development pipeline that experimented with isolating problem labels from the training set
* **run_multilayer.py**: Development script for experimenting with the Multilayer Perceptron deep learning model
* **run_rf.py**: Development script for experimenting with and parameter tuning Random Forest Regressor
