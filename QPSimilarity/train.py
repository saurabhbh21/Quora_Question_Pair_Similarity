import os 
import numpy as np 
import tensorflow as tf

from QPSimilarity.dataset import Dataset
from QPSimilarity.utils import PretrainModelUtils
from QPSimilarity.preprocess import Preprocess
from QPSimilarity.neuralnet import CreateModel
from QPSimilarity.constants import Constant

tf.enable_eager_execution()
constant = Constant()

class Train:
    def __init__(self, word_to_index, word_to_vec_map, features, target, dataset_dir = constant.dataset_dir, dataset_filename = constant.dataset_filename):
        self.dataset = Dataset(dataset_dir, dataset_filename, features, target)
        self.model = CreateModel()
        self.word_to_index = word_to_index
        self.word_to_vec_map = word_to_vec_map

    def trainModel(self, split_ratio=0.99, num_epochs=6):
        x_train, y_train, x_test, y_test  = self.dataset.readandSplitDataset(split_ratio)

        preprocess = Preprocess(self.word_to_index)
        x_index_train = np.apply_along_axis(preprocess.sentenceToIndices, 1, x_train)
        x_index_test = np.apply_along_axis(preprocess.sentenceToIndices, 1, x_test)

        
        with tf.device('/CPU:0'):
            classification_model = self.model.rawModel(self.word_to_vec_map, self.word_to_index)
        
        cp_callback = self.model.createCheckpoint()
    
        classification_model.fit(
            x_index_train, y_train, batch_size=256, epochs=num_epochs, verbose=1, 
            callbacks= [cp_callback], 
            validation_data= (x_index_test, y_test)
            )

        classification_model.summary()

        # y_test_predict = classification_model.predict(x_index_test)
        # print('Labels shape:', y_test.shape)
        # print('Predict shape:', y_test_predict.shape)
        # f1_score_validation = tf.contrib.metrics.f1_score(y_test.flatten(), y_test_predict.flatten())
        
        loss_and_metric = classification_model.evaluate(x_index_train, y_train)
        print('Loss and metric:', loss_and_metric)
        
        loss = loss_and_metric[0]
        metric = loss_and_metric[1]

        checkpoint_dir = os.path.dirname(constant.checkpoint_path)
        trained_model_path = tf.train.latest_checkpoint(checkpoint_dir)
        print('Model Path String::', trained_model_path)

        return metric



# dataset_dir = constant.dataset_dir
# dataset_filename = constant.dataset_filename

# features = [value for key , value in constant.features.items()]
# target = [value for key , value in constant.labels.items()]

# pretrained = PretrainModelUtils()
# word_to_index, _, word_to_vec_map = pretrained.readPretrainedModel(glove_path=constant.glove_path)

# training =  Train(dataset_dir, dataset_filename, features, target, word_to_index, word_to_vec_map)
# training.train()
