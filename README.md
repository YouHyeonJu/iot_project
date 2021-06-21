# 라즈베리파이4 를 이용한 코로나 열감지,마스크 감지 프로그램
 누구나 따라 할 수 있는 마스크 감지 프로그램입니다.
## 프로젝트 설명 및 내용
 카메라로 마스크를 쓰지 않는 사람들을 **부저와 적색 LED**로 알려주고, 

 **온도가 37.5°C 이상**이 되면 **적색등과 부저**가 울리게 하는 프로젝트입니다.  
 
 ## 하드웨어 설계 
 
 ![하드웨어 설계 윗판](/hard1.jpg)  
 ![하드웨어 설계](/hard2.jpg)  
 
## 온도 측정과 부저 및 LED
```python
from Adafruit_AMG88xx import Adafruit_AMG88xx
import pygame
import os
import math
import time
import RPi.GPIO as GPIO
import numpy as np
from scipy.interpolate import griddata
from gpiozero import Buzzer

from colour import Color
red=14
green=15
buzzer = Buzzer(18)
GPIO.setmode(GPIO.BCM)
GPIO.setup(red,GPIO.OUT)
GPIO.setup(green,GPIO.OUT)



#low range of the sensor (this will be blue on the screen)
MINTEMP = 26

#high range of the sensor (this will be red on the screen)
MAXTEMP = 32

#how many color values we can have
COLORDEPTH = 1024

os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.init()

#initialize the sensor
sensor = Adafruit_AMG88xx()

points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

#sensor is an 8x8 grid so lets do a square
height = 480
width = 480

#the list of colors we can choose from
blue = Color("indigo")
colors = list(blue.range_to(Color("red"), COLORDEPTH))

#create the array of colors
colors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in colors]

displayPixelWidth = width / 30
displayPixelHeight = height / 30

lcd = pygame.display.set_mode((width, height))

lcd.fill((255,0,0))

pygame.display.update()
pygame.mouse.set_visible(False)

lcd.fill((0,0,0))
pygame.display.update()

#some utility functions
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#let the sensor initialize
time.sleep(.1)
	
while(1):

	#read the pixels
	pixels = sensor.readPixels()
	#print(sensor.readPixels())
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
	pixels = [map(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels]
	
	#perdorm interpolation
	bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
	
	#draw everything
	for ix, row in enumerate(bicubic):
		for jx, pixel in enumerate(row):
			pygame.draw.rect(lcd, colors[constrain(int(pixel), 0, COLORDEPTH- 1)], (displayPixelHeight * ix, displayPixelWidth * jx, displayPixelHeight, displayPixelWidth))
	
	pygame.display.update()

```
 
## 마스크 착용여부 감지 모델

