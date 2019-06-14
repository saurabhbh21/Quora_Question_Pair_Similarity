import os 
import numpy as np
import tensorflow as tf

from QPSimilarity.constants import Constant

tf.enable_eager_execution()
constant = Constant()

class EmbeddingLayer:

    def __init__(self, *args, **kwargs):
        pass
    
    @staticmethod
    def pretrainedEmbeddingLayer(word_to_vec_map, word_to_index):
        vocab_len = len(word_to_index) + 1                  
        emb_dim = word_to_vec_map["cucumber"].shape[0]     
    
        emb_matrix = np.zeros((vocab_len, emb_dim))
    
    
        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map[word]

        embedding_layer = tf.keras.layers.Embedding(vocab_len, emb_dim, trainable=False, mask_zero= True)
   
        embedding_layer.build((None,))
    
        embedding_layer.set_weights([emb_matrix])
    
        return embedding_layer


class NeuralNetwork(tf.keras.Model):
    def __init__(self, word_to_vec_map, word_to_index):
        super(NeuralNetwork, self).__init__()

        self.embedding_layer = EmbeddingLayer.pretrainedEmbeddingLayer(word_to_vec_map, word_to_index)
        
        self.lstm_layer1 = tf.keras.layers.LSTM(128, return_sequences = True)
        self.dropout = tf.keras.layers.Dropout(0.5)
        
        self.lstm_layer2 = tf.keras.layers.LSTM(128, return_sequences = False)
        
        self.dense = tf.keras.layers.Dense(128)
        
        self.dense_combined = tf.keras.layers.Dense(512, activation='relu')
        self.classification = tf.keras.layers.Dense(1, activation='sigmoid')

    
    def call(self, inputs):
        x = self.embedding_layer(inputs[:, 0, :])            #For Question1
        x = self.lstm_layer1(x)
        x = self.dropout(x)
        x = self.lstm_layer2(x)
        x = self.dropout(x)
        x = self.dense(x)
        
        y = self.embedding_layer(inputs[:, 1, :])            #For Question2
        y = self.lstm_layer1(y)
        y = self.dropout(y)
        y = self.lstm_layer2(y)
        y = self.dropout(y)
        y = self.dense(y)
        
        z = tf.math.abs(tf.subtract(x, y))
        z = self.dense_combined(z)
        z = self.classification(z)       
                
        return z 

class CreateModel(NeuralNetwork):
    def __init__(self, checkpoint_path=constant.checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(checkpoint_path)

    
    def createCheckpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1,
            save_weights_only=True, save_best_only=True,  #save best weights
            period=1                                      #every epoch
            )

        return cp_callback
        
    
    
    def rawModel(self, word_to_vec_map, word_to_index):
        encoding_model = NeuralNetwork(word_to_vec_map, word_to_index)
        encoding_model.compile(loss = 'binary_crossentropy', optimizer = tf.train.AdamOptimizer(0.0025), metrics=['accuracy'])

        return encoding_model

    


    