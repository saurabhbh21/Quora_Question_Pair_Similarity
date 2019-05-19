import os
import pandas as pd

class Dataset():
    def __init__(self, path, filename, features, target):
        self.filepath = path + os.sep + filename
        self.features = features
        self.target = target
    

    def readandSplitDataset(self, train_ratio):
        dataset = pd.read_csv(self.filepath, keep_default_na=False)
        
        question_pair = pd.DataFrame(dataset, columns=self.features)
        labels = pd.DataFrame(dataset, columns=self.target)

        feature_array = question_pair.to_numpy()
        label_array = labels.to_numpy()

        return self.split_dataset(feature_array, label_array, train_ratio)

    

    def split_dataset(self, feature_array, target_array, train_ratio):
        assert feature_array.shape[0] == target_array.shape[0], 'different num rows in feature and target'
    
        num_rows_split = int(feature_array.shape[0] * train_ratio)
        
        feature_train = feature_array[:num_rows_split]
        feature_test = feature_array[num_rows_split:]
        
        target_train = target_array[:num_rows_split]
        target_test = target_array[num_rows_split:]
        
        return feature_train, target_train, feature_test, target_test
