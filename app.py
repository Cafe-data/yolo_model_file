import io
import json
import os
import cv2
import pathlib
import sys
import torch
import numpy as np
import time
import joblib
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from PIL import Image
from flask import Flask, jsonify, request
from torchvision import transforms



app = Flask(__name__)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
sys.path.insert(0, './yolov5')

model = joblib.load('./data.pkl')

cred = credentials.Certificate('emptydesk-d8a8d-firebase-adminsdk-3rjtx-bc7b07c657.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://emptydesk-d8a8d-default-rtdb.firebaseio.com/'
})
ref = db.reference()


model.eval()      


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    data = np.array(data['images'], dtype=np.uint8)
    results = model(data)
    model_out = results.pandas().xyxy[0]
    total = model_out['name'].count()
    empty_seat_num = model_out['name'].value_counts()['empty']
    ref.update({'ediya':{'total': total, "empty":empty_seat_num}})
    return {'total': str(total), 'empty': str(empty_seat_num)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)