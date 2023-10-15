import cv2
import numpy as np
# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names (80 classes, including "person")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize the camera
cap = cv2.VideoCapture(0)



# Initialize variables for counting people
count = 0 
prev_count = 0
while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1920,1080))
    cv2.imshow('FRAME',frame)
    if not ret:
        break
    

    height, width = frame.shape[:2]

    # Create a blob from the frame and set it as input to the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass to get the output layers
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to "person"
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
         
        if i in indexes:
            count += 1
            x, y, w, h = boxes[i]
            
            # Draw a bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            

    # Display the count on the frame
    cv2.putText(frame, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('People Counter', frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()