# Disaster Response Pipeline Project

## Abstract

This project is a demonstration of a complete data science solution. From converting raw data into a usable format, training a classification model, to displaying actionable results in a web app. The dataset, provided by FigureEight, contains messages and various emergency categories that they belong to. The idea is to classify messages into certain buckets so various types of aid could be delivered appropriately. 

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files

### app/run.py

This script starts the flask web app and provides the charts that appear.

### app/templates/go.html

This template displays the result of a message query's classification.

### app/templates/master.html

This template is the main page of the web app and contains the charts generated in `run.py` and search bar. 

### data/process_data.py

A script to take the existing CSV files, clean them, prepares the data for machine learning, and saves the result to a local database file.

### data/disaster_categories.csv

For each ID, the previously tagged categorical results are listed. This is for the training the model's output.

### data/disaster_messages.csv

For each ID, the message content is listed. This data will be cleaned, tokenized and combined with the output categories.

### models/train_classifier.py

A script to take the output SQL table from `data/process_data.py`  and trains a model to process messages. 
