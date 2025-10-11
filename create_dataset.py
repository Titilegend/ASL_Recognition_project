import mediapipe as mp
import cv2
import os
import pickle
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True,min_detection_confidence=0.3)

directory = 'Data/'
data = []
labels =[]
for dir in os.listdir(directory):
    dir_path = os.path.join(directory,dir)
    if not os.path.isdir(dir_path):
        continue
    print(f"processing{dir}...")
    for img_path in os.listdir(dir_path):
        data_aux = []
        img = cv2.imread(os.path.join(directory,dir,img_path))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir)

f = open("data_pickle", 'wb')
pickle.dump({'data':data, 'labels':labels},f)
f.close()




    

print("Label names found:",sorted(set(labels)))
print("Total samples collected:",len(data))


print("Samples per class:")
for label in sorted(set(labels)):
    count = labels.count(label)
        
    print(f"{label}:{count} samples")






                   