import numpy as np
import cv2
import os
import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help = r"C:\Users\user\Desktop\YOLO_OD\images" )
ap.add_argument("-y", "--yolo", required=True, help = r"C:\Users\user\Desktop\YOLO_OD\yolo-coco")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak predictions")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima supression")

args = vars(ap.parse_args())

#Loading classes names on which our data is trained on
labelspath = os.path.sep.join([args['yolo'],'coco.names'])
labels = open(labelspath).read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

#Loading YOLO model weights and architecture configuration
weightspath = os.path.sep.join([args['yolo'],'yolov3.weights'])
configpath = os.path.sep.join([args['yolo'],'yolov3.cfg'])

#Using opencv to load YOLO object detector
net = cv2.dnn.readNetFromDarknet(configpath,weightspath)

#Loading input image 
img = cv2.imread(args['image'])
(H,W) = img.shape[:2]

#determine only *output* layer names that is required from YOLO
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Constructing blob from image and then perform a forward pass of YOLO object detector 
# Will give us bounding boxes and associated probabilties
blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
net.setInput(blob)
#start = time.time()
layerOutputs = net.forward(ln)
#end = time.time()

# show timing information on YOLO
#print("[INFO] YOLO took {:.6f} seconds".format(end - start))

#Initializing list for bounding boxes , confidences , class IDs
boxes , confidences , classIDs = [] , [] , []

#Looping over each of outputlayers
for output in layerOutputs:
    #Looping over each of the detections
    for detection in output:
        #Getting classID and confidence of current detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if (confidence > args['confidence']):
            
            
            # scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
            
            box = detection[0:4] * np.array([W, H, W, H])
            
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
            
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
			# and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)


# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

#Ensuring one box detection
if (len(idxs) > 0):
    for i in idxs.flatten():
        #Extract bounding box coordinates
        (x,y) = (boxes[i][0] , boxes[i][1])
        (w,h) = (boxes[i][2] , boxes[i][3])

        #Draw bounding boxes 
        color = [int(c) for c in colors[classIDs[i]]]
        cv2.rectangle(img , (x,y) , (x+w,y+h) , color , 2 )

        text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


# Show the output Image
cv2.imshow("Image",img)
cv2.waitKey(0)






