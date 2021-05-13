from threading import Thread
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request
import numpy as np
import tensorflow as tf
import imutils
import json
import math
import cv2
import os

from tensorflow.python.keras.backend import dtype

class npEncoder(json.JSONDecoder):
  def default(self, obj):
    if isinstance(obj, np.int32):
      return int(obj)
    return json.JSONDecoder.default(self,obj)
  
print("얼굴 감지 모델 로딩중...")
prototxtPath = 'D:\\App\\face_detector\\deploy.prototxt'
weightPath = 'D:\\App\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath,weightPath)

print('얼굴 마스크 감지 모델 로딩중...')
maskNet = load_model('D:\\App\\mask_detector.model')


def detect_and_predict_mask(frame):
  (h,w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

  faceNet.setInput(blob)
  detections = faceNet.forward()

  faces = []
  locs = []
  preds = []

  for i in range(0, detections.shape[2]):
    confidence = detections[0,0,i,2]

    if confidence > 0.5:
      box = detections[0,0,i,3:7] * np.array([w,h,w,h])
      (startX, startY, endX, endY) = box.astype("int")

      (startX, startY) = (max(0, startX), max(0, startY))
      (endX, endY) = (min(w-1, endX), min(h-1,endY))

      face = frame[startY:endY, startX:endX]
      face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
      face = cv2.resize(face, (224, 224))
      face = img_to_array(face)
      face = preprocess_input(face)

      faces.append(face)
      locs.append((startX, startY, endX, endY))
  
  if len(faces) > 0:
    faces = np.array(faces, dtype="float32")
    preds = maskNet.predict(faces, batch_size=32)
  else:
    preds = 0
  return(locs, preds)


app = Flask(__name__)

@app.route('/processing',methods=['POST'])
def Processing():
  frame = request.json
  frame = np.array(frame['frame'],dtype='uint8').reshape(300,400,3)
  (locs, preds) = detect_and_predict_mask(frame)
  labels = []
  try:
    for ind_x,val_x in enumerate(preds):
      if preds[ind_x][0] > preds[ind_x][1]:
        labels.append("Mask")
      else:
        labels.append("No Mask")

  except:
    pass
  locs = np.array(locs,dtype='uint8')
  locs = locs.tolist()
  result = {'locs':locs, \
            'labels':labels}
  
  print(result)
  
  return (result)


if __name__=='__main__':
  app.run(host='0.0.0.0',port=1022)