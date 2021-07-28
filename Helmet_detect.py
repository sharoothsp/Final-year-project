import sys
sys.path.insert(1,'/Users/user/anaconda3/Lib/site-packages')
import cv2
import numpy as np






def numberp(imgs):
    face_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Read the input image
#img = cv2.imread('img100.jpg')
    img=imgs
# Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect numberplate
    faces = face_cascade.detectMultiScale(gray, 1.05, 4)
#faces = face_cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 7)

# Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

def helmet(images):
    f=0
    net = cv2.dnn.readNet("yolov3-helmet.weights", "yolov3-helmet.cfg")
    classes = []
    with open("helmet.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    img = images 
    #img = cv2.imread("img2t1.png")
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
            # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    color = (255, 0, 0)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
       
            if label == 'Helmet':
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                f=1
            
            #cv2.putText(img, label, (x, y + 30), font, 1, color, 3)

   
    


    #cv2.imshow("Image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return f
    

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
bike= []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]



layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread("tp6.jpeg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape


blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        
        confidence = scores[class_id]
        
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
            
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
color = (255, 0, 0)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        
        if label == 'motorbike':
            
            z=int(y-(h*0.5))
            if z<0:
                z=0
            cv2.rectangle(img, (x, z), (x + w, y + h), color, 2)
            
            newimg = img[z:y+h,x:x+w]
            pri=helmet(newimg)
            
            
            if pri==1:
                pass
            else:
                print("detected")
                numberp(newimg)
            #cv2.putText(newimg, label, (x+w-10, y+h-10), font, 1, color, 3)
            #numplate(newimg)
#print(height,width)
if height > 700:
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

