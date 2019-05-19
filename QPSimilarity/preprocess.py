import numpy as np 
import spacy


class Preprocess:
    def __init__(self, word_to_index):
        self.word_to_index = word_to_index
        self.nlp = spacy.load('en')

    def sentenceToIndices(self, sentences, max_len=500):

        
        indices_vector = np.zeros(shape=(sentences.shape[0], max_len))
        for index in range(sentences.shape[0]):
            sentence_words =   sentences[index].split() #[str(token.text).lower() for token in self.nlp(sentences[index])]

            vectorize_indices = np.vectorize(self.word_to_index.get, otypes = [int])
            indices =   vectorize_indices(sentence_words, 0)
            
            ##extend length of indices with vector of length 500
            indices = np.pad(indices, (0, max_len - indices.shape[0]), 'constant')
            #print(indices.shape)
            indices_vector[index] = indices


        return indices_vector