import requests

if __name__ == "__main__":
    question1 = 'What is difference between AI and ML'
    question2 = 'How AI is different from ML'

    data = {'question1': question1, 'question2':question2}
    url = 'http://127.0.0.1:5000/predict'

    response =  requests.post(url, json=data)

    print(type(response.text))