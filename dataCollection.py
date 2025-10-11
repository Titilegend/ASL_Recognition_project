import cv2
import os
import pickle
#import mediapipe as mp
import matplotlib.pyplot as plt
#start video capture



#directory for storing images

directory = 'Data/'
if not os.path.exists(directory):
    os.makedirs(directory)

number_of_classes = 34
dataset_size = 800


cap = cv2.VideoCapture(0)
class_names = [
   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',"Hello","Goodbye","help","Home","ThankYou","IloveYou","Yes","No"
]


for j in range(len(class_names)):
    class_name = class_names[j]
    class_dir = os.path.join(directory,class_name)
    if not os.path.exists(class_dir):
       os.mkedirs(class_dir)

    print('Collecting data for class {}'.format(class_name))
    done=False
    while True:
        ret,frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame,1)

        cv2.putText(frame,'Ready?  Press "Q" ! :)',(100,50),cv2.FONT_HERSHEY_COMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)
        cv2.putText(frame, 'Class: {}'.format(class_name), (100, 100), 
                   cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        key =  cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key ==27:
            print('Skipped class:{}'.format(class_name))
            done=True
            break

    if done:
        continue




    counter = 0
    while counter < dataset_size:
        ret,frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame,1)
        cv2.putText(frame, 'Collecting: {}'.format(class_name), (50, 50), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Progress: {}/{}'.format(counter, dataset_size), (50, 100), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow('frame',frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir,'{}.jpg'.format(counter)),frame)

        counter +=1

          # Allow skipping with ESC key
        if cv2.waitKey(25) == 27:
            print('Skipped class: {}. Collected {} samples.'.format(class_name, counter))
            break

    print('Completed class: {} - {}/{} samples'.format(class_name, counter, dataset_size))

cap.release()
cv2.destroyAllWindows()

print("Data collection completed!")
print("Total classes: {}".format(number_of_classes))
print("Target samples per class: {}".format(dataset_size))
print("Expected total: {} samples".format(number_of_classes * dataset_size))

    # dictionary to count number of images in each subdir
"""
    count = {
        'a':len(os.listdir(directory +'/A')),
        'b':len(os.listdir(directory +'/B')),
        'c':len(os.listdir(directory +'/C')),
        'd':len(os.listdir(directory +'/D')),
        'e':len(os.listdir(directory +'/E')),
        'f':len(os.listdir(directory +'/F')),
        'g':len(os.listdir(directory +'/G')),
        'h':len(os.listdir(directory +'/H')),
        'i':len(os.listdir(directory +'/I')),
        'j':len(os.listdir(directory +'/J')),
        'k':len(os.listdir(directory +'/K')),
        'l':len(os.listdir(directory +'/L')),
        'm':len(os.listdir(directory +'/M')),
        'n':len(os.listdir(directory +'/N')),
        'o':len(os.listdir(directory +'/O')),
        'p':len(os.listdir(directory +'/P')),
        'q':len(os.listdir(directory +'/Q')),
        'r':len(os.listdir(directory +'/R')),
        's':len(os.listdir(directory +'/S')),
        't':len(os.listdir(directory +'/T')),
        'u':len(os.listdir(directory +'/U')),
        'v':len(os.listdir(directory +'/V')),
        'w':len(os.listdir(directory +'/W')),
        'x':len(os.listdir(directory +'/X')),
        'y':len(os.listdir(directory +'/Y')),
        'z':len(os.listdir(directory +'/Z')),
        'space':len(os.listdir(directory +'/Space')),
        'nothing':len(os.listdir(directory +'/Nothing')),
        'delete':len(os.listdir(directory +'/Delete')),
    }

    # get dimension for captured frame

    row = frame.shape[1]
    col = frame.shape[0]


    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)


    cv2.imshow("data",frame)
    cv2.imshow("ROI", frame[40:400,0:300]) """