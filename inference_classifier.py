import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter

model_dict = pickle.load(open('asl_alphabet_model.p','rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True,min_detection_confidence=0.3,max_num_hands=1)

labels_dict = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',
    13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z',
    26:"Hello",27:"Goodbye",28:"help",29:"Home",30:"ThankYou",31:"IloveYou",32:"Yes",33:"No"
}

prediction_history = []
SMOOTHING_WINDOW = 5

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret,frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame,1)

    H,W,_ = frame.shape


    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
       # for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    #for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
            x_.append(x)
            y_.append(y)
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]


            prediction_history.append(predicted_character)
            if len(prediction_history) > SMOOTHING_WINDOW:
                prediction_history.pop(0)
            



            if len(prediction_history)>=3:
                smoothed_prediction = Counter(prediction_history).most_common(1)[0][0]
            else:
                smoothed_prediction = predicted_character




            


            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20


            x2 = int(max(x_) * W) +20
            y2 = int(max(y_) * H) +20


            x1 = max(0,x1)
            y1 = max(0,y1)
            x2= min(W,x2)
            y2 = min(H,y2)


        # prediction= model.predict([np.asarray(data_aux)])
            #predicted_character = prediction[0]


        

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
            cv2.putText(frame,f"Current:{predicted_character}",(x1,y1-60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.putText(frame,f"Smoothed:{smoothed_prediction}",(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame,smoothed_prediction,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)


            hand_width = x2 - x1
            if hand_width <100:
                cv2.putText(frame,"MOVE CLOSER",(50,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            elif hand_width >300:
                cv2.putText(frame,"MOVE FARTHER",(50,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)



    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()