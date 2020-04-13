from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.layers import Dropout

class createmodel:
    def __init__(self):
        self._name = 'name'
        self._model = None
        
    def create_model(self, vocab_size, num_tags):
        model = Sequential()
        model.add(Dense(50, input_shape=(vocab_size,), activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(num_tags, activation='sigmoid'))
    
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
        return model
    
    def get_model_summary(self, model):
        model.summary()
        
        
    def fit_model(self, model, train_data, train_tags):
        self._model = model
        self._model.fit(train_data, train_tags, epochs=2, batch_size=128, validation_split=0.2)
        
    
    def evaluate_model(self, test_data, test_tags):
        return self._model.evaluate(test_data, test_tags, batch_size=128)
    
    
    def save_model(self, model_name):
        self._model.save(model_name + '.h5')

