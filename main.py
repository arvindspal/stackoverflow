import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
import pickle
import json
import math

#load custom classes..
from data.loaddata import loadData
from data.processdata import process_text
from model.createmodel import createmodel
from prediction.predictionclass import prediction
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)

encoder = None


@app.route('/')
def index():    
    name = 'name' 
    load_query()
    res=['load kar', 'jaldi kar']
    return render_template('index.html', prediction_text='Message $ {}'.format(res[0]))
 
    
@app.route('/question')
def question():
    return render_template('prediction.html')


#@app.route('/question')
def predict_tags():
    processor_name = 'stackoverflowquestions'
    model_name = 'stackoverfowmodel'
    
    q = ['how to preprocess keras model in lambda layer using jupyter notebook in python?']
    
    prediction_obj = prediction()
    
    prediction_obj.from_path('./',model_name, processor_name)
    predictions = prediction_obj.predict(q)
    
    #encoder = MultiLabelBinarizer()
    
    tags_list = []
    
    for i in range(len(predictions)):
        for idx, val in enumerate(predictions[i]):
            if val > 0.1:
                tags_list.append(encoder.classes_[idx])
                #tags_list.append(val)

    return render_template('prediction.html', tags=tags_list)
    #return jsonify(tags_list)


def encode_tags(df):
    split_tags = [tags.split(',') for tags in df['tags'].values]
        
    encoder = MultiLabelBinarizer()
    encoded_tags = encoder.fit_transform(split_tags)
        
    return encoded_tags

       
@app.route('/model')   
def prepare_model():
    VACAB_SIZE = 400
    processor_name = 'stackoverflowquestions'
    model_name = 'stackoverfowmodel'
    query = load_query()
    
    loaddata_obj = loadData(query)
    
    #load data
    df = loaddata_obj.get_data()
    
    ## get encoded tags..
    #encoded_tags = loaddata_obj.encode_tags(df)
    encoded_tags = encode_tags(df)
    
    ## get the size of the df
    size = math.ceil(0.8*len(encoded_tags))
    
    ## tags - train and test split
    train_tags, test_tags = loaddata_obj.tags_train_test_split(encoded_tags, size)
    
    ## train and test split of questions..
    train, test = loaddata_obj.text_train_test_split(df, size)
    
    ## get processor..
    processer = process_text(VACAB_SIZE)
    
    ##transform train test data..
    train_data, test_data = loaddata_obj.transform_train_test_data(VACAB_SIZE, processer, train, test)
    
    ## save processor for future use..
    loaddata_obj.save_processor(processer, processor_name)
    
    
    ## now create the model..
    num_tags = len(encoded_tags[0])
    
    createmodel_obj = createmodel()
    model = createmodel_obj.create_model(VACAB_SIZE, num_tags)
    
    ## get model summary..
    createmodel_obj.get_model_summary(model)
    
    ## fit the model..
    createmodel_obj.fit_model(model, train_data, train_tags)
    
    ## evaluate model..
    res = createmodel_obj.evaluate_model(test_data, test_tags)
    
    if (res[1] > 0.8):
        ##save model..
        createmodel_obj.save_model(model_name)
        
    return render_template('index.html', prediction_text='Accuracy $ {}'.format(res[1]))


def load_query():
    path = 'query.txt'
    f = open(path, "r")
    if f.mode == 'r':
        return f.read()
    else:
        'file not found'
        

if __name__ == '__main__':
	#app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
