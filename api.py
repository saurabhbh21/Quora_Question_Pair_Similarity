import numpy  as np 
from flask import Flask, request, jsonify

from QPSimilarity.constants import Constant
from QPSimilarity.utils import PretrainModelUtils
from QPSimilarity.predict import Predict
from QPSimilarity.train import Train



constant = Constant()

pretrained = PretrainModelUtils()
word_to_index, index_to_word, word_to_vec_map = pretrained.readPretrainedModel(glove_path=constant.glove_path)

app = Flask(__name__)

@app.route("/")
def default():
    return 'Please use right end point'


@app.route("/predict", methods=['POST'])
def predict_api():
    data = request.json
    
    question1 = data['question1']
    question2 = data['question2']
    question_array = np.array([question1, question2]).reshape(1,2)
    
    predict = Predict(word_to_index, word_to_vec_map)
    prediction = predict.predictOnTest(question_array)
    
    data['result'] = str(prediction[0][0])
    result_json = jsonify(data)

    return result_json

@app.route("/train", methods=['GET','POST'])
def train_api():
    train_metric = request.json
    
    trainset_split = train_metric['split']
    num_epochs = train_metric['epochs']

    features = [value for key , value in constant.features.items()]
    target = [value for key , value in constant.labels.items()]

    training =  Train(word_to_index, word_to_vec_map, features=features, target=target)
    
    validation_res = training.trainModel(trainset_split, num_epochs)

    result = {'validation_accuracy': str(validation_res)}

    return jsonify(result)




if __name__ == "__main__":
    app.run()
