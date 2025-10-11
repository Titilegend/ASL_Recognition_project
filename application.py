from audioop import avg
import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer,VideoProcessorBase


st.set_page_config(page_title="ASL Recognition", layout = "wide")

@st.cache_resource

def load_model():
    model_dict = pickle.load(open('asl_alphabet_model.p','rb'))
    return model_dict['model']

model = load_model()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True,min_detection_confidence=0.3,max_num_hands=1)

labels_dict = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',
    13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z',
    26:"Hello",27:"Goodbye",28:"help",29:"Home",30:"ThankYou",31:"IloveYou",32:"Yes",33:"No"
}

#app

st.title("American Sign Language Recognition")

st.markdown("""
Gesture ASL signs to your webcam and see them recognized in real-time!"""
)

tab1,tab2 = st.tabs(["ASL Recognition","Word Model"])


with tab1:

    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Detection Confidence",0.1,1.0,0.3)
    smoothing_window = st.sidebar.slider("Smoothing Window",1,10,5)



    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []




    class ASLVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.prediction_history = []



        def recv(self,frame):
            img = frame.to_ndarray(format="bgr24")

            img = cv2.flip(img,1)

            H,W,_ = img.shape


            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)


            data_aux = []
            x_ = []
            y_ = []

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
            # for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
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


                self.prediction_history.append(predicted_character)
                if len(self.prediction_history) > smoothing_window:
                    self.prediction_history.pop(0)
                



                if len(self.prediction_history)>=3:
                    smoothed_prediction = Counter(self.prediction_history).most_common(1)[0][0]
                else:
                    smoothed_prediction = predicted_character


                st.session_state.current_prediction = predicted_character
                st.session_state.smoothed_prediction = smoothed_prediction
                st.session_state.prediction_history = self.prediction_history.copy()




            


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


        

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),4)
                #cv2.putText(img,f"Current:{predicted_character}",(x1,y1-60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
                #cv2.putText(img,f"Smoothed:{smoothed_prediction}",(x1,y1-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.putText(img,smoothed_prediction,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,1.3,(0,255,0),3,cv2.LINE_AA)


                hand_width = x2 - x1
                if hand_width <100:
                    cv2.putText(img,"MOVE CLOSER",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                elif hand_width >300:
                    cv2.putText(img,"MOVE FARTHER",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            return av.VideoFrame.from_ndarray(img,format="bgr24")

    col1,col2 = st.columns([2,1])


with col1:
    st.header("Live Camera Feed")
    st.markdown("position your hand in the frame and show asl letters")



    webrtc_ctx = webrtc_streamer(
        key = "asl-recognition",
        video_processor_factory = ASLVideoProcessor,
        rtc_configuration = {"iceServers": [{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints =  {"video":True,"audio":False},
    )


with col2:

    st.header("Recognition Results")


    if 'current_prediction' in st.session_state:
        st.subheader("Current Prediction")

        st.markdown(f"## {st.session_state.current_prediction}")
        st.subheader("Smoothed History")

        st.markdown(f"## {st.session_state.smoothed_prediction}")

        st.subheader("Prediction History")


        history_text = "->".join(st.session_state.prediction_history[-10:])
        st.code(history_text)

    else:
        st.info("Show your hand to the camera to see predictions!")



    st.header("Instructions")
    st.markdown("""
      1. Make sure your hand is clearly visible
        2. Show one ASL letter at a time
        3. Keep your hand steady for better recognition
        4. Use proper lighting""")


with tab2:
    #word model

    st.header("ASL Word Recognition")
    st.info("Word Model Under Development")


