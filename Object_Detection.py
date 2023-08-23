import cv2 as cv
import numpy as np

threshold = 0.6   #Threshold to detect objects

nms_threshold = 0.35     # Non-Maximum Suppression, used to deal with duplicate bounds. 
                        # Suppresses boxes with non-maximum confidence levels.
                        # 1 = minimal supression, as # is smaller suppression is greater

#Taking Webcam Video for input
#webcam = cv.VideoCapture(0) #Using Internal Webcam
webcam = cv.VideoCapture(1, cv.CAP_DSHOW) #Changes API for faster response when using external webcam
webcam.set(3, 1500) #Width settings
webcam.set(4, 900)  #Height settings
webcam.set(10, 150) #Brightness settings


#Names of Classes in coco.names
classFile = 'Proof Of Concept\coco.names'

#Paths to Model Files
configFile = 'Proof Of Concept\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsFile = 'Proof Of Concept\\frozen_inference_graph.pb'

#Reading the class names from coco.names
classNames = []
#using 'with' ensures that file is closed when read is complete
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)  

#Instatiating Detection Model
net = cv.dnn_DetectionModel(weightsFile, configFile)

#Default Parameters in  Model Documentation
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = webcam.read() #Reading Webcam
    classIDs, confidenceLevel, boundBox = net.detect(img, confThreshold = threshold) #Detecting the Objects
    #Formatting Parameters to input into dnn.NMSBoxes Method
    boundBox = list(boundBox)  #Converting boundBox from numpy array to list
    confidenceLevel = list(np.array(confidenceLevel).reshape(1,-1)[0]) #Converting confidenceLevels to a list of floats
    confidenceLevel = list(map(float, confidenceLevel))

    #This method looks at bounding boxes with the indicies (that mark object class) and will get rid of duplicates
    indicies = cv.dnn.NMSBoxes(boundBox,confidenceLevel,threshold, nms_threshold)
    print(indicies)

    for i in indicies:
        box = boundBox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv.putText(img,classNames[classIDs[i]-1].upper(),(box[0]+10,box[1]+30),
        cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        #Display Confidence Level Code (Work in Progress)
        cv.putText(img, str(round(confidenceLevel[i]* 100,2)) + " %",(box[0]+300, box[1] + 30), 
        cv.FONT_HERSHEY_COMPLEX,1, (0, 255, 0), 2)

    #Code For Displaying Output
    cv.imshow('Output', img)

    #Terminating the Program:
    #Code for closing the Window ("Hit q on keyboard")
    if (cv.waitKey(1) & 0xFF) == ord("q"):
        cv.destroyAllWindows()
        break
    #Code for closing the Window ("Hit 'X' on GUI")
    if cv.getWindowProperty('Output',cv.WND_PROP_VISIBLE) < 1:        
        cv.destroyAllWindows()
        break     
    
    
    

    
    