## Description
A RESTful application for training and prediction on whether two Question have similar context (i.e. asking for similar information)

## Dataset
Quora Question Pair Similarity dataset from the competition posted on Kaggle is used for training the model

## How To Train and Predict
main_train.py:  
Python file sending json request for training the model for 'Whether Pair of Questions have same Context' and receiving validation accuracy as json

main_predict.py: 
Python file sending json request for predicting the 'Whether Pair of Questions have same Context' and receiving result as same/ different (1/0) as json.

