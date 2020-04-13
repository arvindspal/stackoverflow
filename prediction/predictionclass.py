import os
import pickle
import tensorflow.keras as keras


class prediction:
    def __init__(self):
        self._model = None
        self._processor = None
        
    def predict(self, instances):
        processed_data = self._processor.transform_text(instances)
        predicted_data = self._model.predict(processed_data)
        
        return predicted_data.tolist()
    
    
    def from_path(self, model_dir, model_name, processor_name):
        pm = 'C://Users//asp//stack-overflow//stackoverflow//' + model_name + '.h5'
        pp = 'C://Users//asp//stack-overflow//stackoverflow//' + processor_name + '.pkl'

        #model = keras.models.load_model(os.path.join(model_dir, model_name + '.h5'))
        self._model = keras.models.load_model(pm)
        
        with open(pp, 'rb') as f:
            self._processor = pickle.load(f)
            
        #return cls(model, processor)