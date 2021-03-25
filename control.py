import RPi.GPIO as GPIO 
import time 
from gpiozero import Buzzer
from Adafruit_AMG88xx import Adafruit_AMG88xx

red=14
green=15
buzzer = Buzzer(18)
GPIO.setmode(GPIO.BCM)
GPIO.setup(red,GPIO.OUT)
GPIO.setup(green,GPIO.OUT)

sensor = Adafruit_AMG88xx()
time.sleep(.1)

while(1):
    pixels=sensor.readPixels()
	print(pixels)
	time.sleep(1)
    if max(pixels)>=37.5:
            print(max(pixels))
            print("warning!")
            GPIO.output(red,True)
            #buzzer
            for i in range(7):
                buzzer.on()
                time.sleep(0.4)
                buzzer.off()
                time.sleep(0.4)
            GPIO.output(red,False)