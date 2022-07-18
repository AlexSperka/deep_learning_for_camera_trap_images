# import cv2
import numpy as np

import requests
import json
import base64

def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')

def visualize_animal_activity(img):
    url = "http://0.0.0.0:5000/model/api/v1.0/"
    headers = {'Content-Type': 'application/json'}
    image_req = json.dumps({'img': 'str(encode_img(img))'})
    # image_req = json.dumps({'img': str(encode_img(img))})
    response = requests.request("POST", url=url+'recognize', headers=headers, data=image_req)
    activity = json.loads(response.content)['activity']
    confidence = json.loads(response.content)['confidence']
    return activity, confidence    

if __name__ == '__main__':
    # img = cv2.imread("C:/Users/alexa/OneDrive/Dokumente/Thesis_MA/Upload 1/Lion/20190112 - CS1_5891")    # Add the image path, ex: '/home/animal.jpg'
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=''
    activity, confidence = visualize_animal_activity(img)  
    print(activity, confidence) 
