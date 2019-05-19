import requests

if __name__ == "__main__":
    num_epoch = 1
    split_ratio = 0.0001

    data = {'split': split_ratio, 'epochs':num_epoch}
    url = 'http://127.0.0.1:5000/train'

    response =  requests.post(url, json=data)

    print(response.text)