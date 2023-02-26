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
from PIL import Image
from flask import Flask, jsonify, request
from torchvision import transforms



app = Flask(__name__)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
sys.path.insert(0, './yolov5')
model = joblib.load('/Users/melod/graduate_/data.pkl')

model.eval()      


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    data = np.array(data['images'], dtype=np.uint8)
    results = model(data)
    model_out = results.pandas().xyxy[0]
    total = model_out['name'].count()
    empty_seat_num = model_out['name'].value_counts()['empty']
   
    return {'total': str(total), 'empty': str(empty_seat_num)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)