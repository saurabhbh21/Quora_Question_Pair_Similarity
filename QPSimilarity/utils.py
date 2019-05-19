import numpy as np

from QPSimilarity.constants import Constant

constant = Constant()

class PretrainModelUtils():
    def __init__(self):
        pass
    

    def readPretrainedModel(self, glove_path=constant.glove_path):
    
        with open(glove_path, 'r') as f:
            words = set()
            word_to_vec_map = {}
            
            for line in f:
                line = line.strip().split()
                
                curr_word = line[0]
                words.add(curr_word)
                
                word_to_vec_map[curr_word] = np.array(line[1:], dtype = np.float32)
                
            
            i = 1
            word_to_index = {}
            index_to_word = {}
            
            for w in sorted(words):
                
                word_to_index[w] = i
                index_to_word[i] = w
                
                i = i + 1
            
        return word_to_index, index_to_word, word_to_vec_map


    







