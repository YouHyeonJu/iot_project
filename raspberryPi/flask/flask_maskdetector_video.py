import cv2
import imutils
from imutils.video import VideoStream
import time
import requests
import numpy as np
import json

#Temperature check lib
from Adafruit_AMG88xx import Adafruit_AMG88xx

#buzzer and led lib
import RPi.GPIO as GPIO
from gpiozero import Buzzer

#led color and buzzer setting
red=14
green=15
buzzer = Buzzer(18)
GPIO.setmode(GPIO.BCM)
GPIO.setup(red,GPIO.OUT)
GPIO.setup(green,GPIO.OUT)

print('starting video stream...')
vs = VideoStream(src=0).start()
sensor = Adafruit_AMG88xx()
time.sleep(2.0)

while True:
    
    # LED GREEN ON
    GPIO.output(green,True)
    
    # videoCapture  use #
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    pixels = np.array(frame)
    
    #Temperature check
    T_pixels = sensor.readPixels()
    max_pixel = max(T_pixels)


    headers = {'Content-Type':'application/json'}
    address = #"http://xxx.xxx.xxx.xxx:port/processing" <- input your pc ip address
    data = {'frame':pixels.tolist()}

    result = requests.post(address, data=json.dumps(data), headers=headers)
    # break
    result = str(result.content,encoding='utf-8')
    result = json.loads(result)
    
    locs = result['locs']
    labels = result['labels']

    for box, label in zip(locs,labels):
        (startX, startY, endX, endY) = box
        
        color = (0,255,0) if label == "Mask" else (0,0,255)

        mark = "{}: {:.2f}.C".format(label, max_pixel)

        #buzzer and led red
        if max_pixel >= 37.5:
            GPIO.output(green,False)
            GPIO.output(red,True)
            #buzzer
            for i in range(7):
                buzzer.on()
                time.sleep(0.4)
                buzzer.off()
                time.sleep(0.4)
            GPIO.output(red,False)
        elif label == "No Mask":
            GPIO.output(green,False)
            GPIO.output(red,True)
            buzzer.on()
            time.sleep(0.4)
            buzzer.off()
            time.sleep(0.4)
            GPIO.output(red,False)

        cv2.putText(frame, mark, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()