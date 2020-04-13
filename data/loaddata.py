import pydata_google_auth
import google.auth
from google.cloud import bigquery
#from google.cloud import storage
import pandas as pd
from sklearn.utils import shuffle
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from data.processdata import process_text


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C://Users//asp//Desktop//googlecloudapi//arvindTest-c1bced9fc3a1.json"

class loadData:
    def __init__(self, query):
        self._query = query
        self._random_state = 22
        
    def get_data(self):
        ## execute the query to fetch the data
        df = pd.read_gbq(self._query, dialect='standard')
        
        ## shuffle the data
        df = shuffle(df, random_state=self._random_state)
        
        return df
    
    
    def encode_tags(self, df):
        split_tags = [tags.split(',') for tags in df['tags'].values]
        
        encoder = MultiLabelBinarizer()
        encoded_tags = encoder.fit_transform(split_tags)
        
        return encoded_tags
    
    def tags_train_test_split(self, encoded_tags, size):
        
        train_tags = encoded_tags[:size]
        test_tags = encoded_tags[size:]
        
        return train_tags, test_tags
    
    
    def text_train_test_split(self, df, size):
        train_title = df['title'].values[:size]
        test_title = df['title'].values[size:]
        
        return train_title, test_title
    
    def transform_train_test_data(self, VACAB_SIZE, processer, train_title, test_title):
        #VACAB_SIZE = 400

        processer.create_tokenizer(train_title)
        train_data = processer.transform_text(train_title)
        test_data = processer.transform_text(test_title)
        
        return train_data, test_data
    
    
    def save_processor(self, processor, processor_name):
        with open('./' + processor_name + '.pkl', 'wb') as f:
            pickle.dump(processor, f)
        
        
        

        