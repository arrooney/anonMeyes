import kivy
kivy.require('1.0.6') # replace with your current kivy version !

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from PIL import Image
import time
import os
import uuid

from pyimagesearch.face_blurring import anonymize_face_pixelate
import numpy as np
import argparse
import cv2

#begin UI Build


Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (1080, 1920)
        play: True
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')

#End UI Build

class CameraClick(BoxLayout):
    def capture(self):
        camera = self.ids['camera']
        #timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("exif.png")
        print("Captured")
        image = Image.open('exif.png')
        data = list(image.getdata())
        image_no_exif = Image.new(image.mode, image.size)
        image_no_exif.putdata(data)
        randomstring = str(uuid.uuid4())
        filename = randomstring + '.png'
        image_no_exif.save(filename)
        os.remove('exif.png')


        prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
        weightsPath = os.path.sep.join(["face_detector",
        	"res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the input image from disk, clone it, and grab dimensions
    
        image = cv2.imread(filename)
        orig = image.copy()
        (h, w) = image.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        	(104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
        	confidence = detections[0, 0, i, 2]

        	# filter out weak detections 
        	if confidence > 0.5:
        		# compute the (x, y)-coordinates of the bounding box for the
        		# object
        		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        		(startX, startY, endX, endY) = box.astype("int")

        		# extract the face ROI
        		face = image[startY:endY, startX:endX]

        		face = anonymize_face_pixelate(face,
        			blocks= 12)

        		# store the blurred face in the output image
        		image[startY:endY, startX:endX] = face

        cv2.imwrite(filename , image)





class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()
