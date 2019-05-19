import os
import numpy as np 
import tensorflow as tf

from QPSimilarity.utils import PretrainModelUtils
from QPSimilarity.neuralnet import CreateModel
from QPSimilarity.preprocess import Preprocess
from QPSimilarity.constants import Constant

tf.enable_eager_execution()

constant = Constant()


class Predict:

    def __init__(self, word_to_index, word_to_vec_map):
        self.word_to_index, self.word_to_vec_map = word_to_index, word_to_vec_map
        
    
    def loadWeights(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest_chkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        model = CreateModel()
        encoding_model =   model.rawModel(self.word_to_vec_map, self.word_to_index)
        encoding_model.load_weights(latest_chkpoint)

        return encoding_model

    
    def predictOnTest(self, question_array, checkpoint_path=constant.checkpoint_path):
        preprocess = Preprocess(self.word_to_index)

        x_index_train = np.apply_along_axis(preprocess.sentenceToIndices, 1, question_array)

        encoding_model = self.loadWeights(checkpoint_path)
        
        output  = encoding_model.predict(x_index_train)
        labels = (output > 0.55).astype(int)
        return labels


# pretrained = PretrainModelUtils()
# word_to_index, _, word_to_vec_map = pretrained.readPretrainedModel(glove_path=constant.glove_path)

# predict = Predict(word_to_index, word_to_vec_map)
# predict.testModel(constant.checkpoint_path)






        
        


