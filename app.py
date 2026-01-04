# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd

from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64



# df = pd.read_csv(r"C:\Users\jwala\OneDrive\Desktop\Emotion-based-music-recommendation-system-main\muse_v3.csv")
df = pd.read_csv("muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
print(df)

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
print(df)

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(emotions):

    data = pd.DataFrame()

    if len(emotions) == 1:
        v = emotions[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'Fearful':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'Happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
    elif len(emotions) == 2:
        times = [30,20]
        for i in range(len(emotions)):
            v = emotions[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':             
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(emotions) == 3:
        times = [55,20,15]
        for i in range(len(emotions)): 
            v = emotions[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':             
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':               
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(emotions) == 4:
        times = [30,29,18,9]
        for i in range(len(emotions)):
            v = emotions[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':               
                data =pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10,7,6,5,2]
        for i in range(len(emotions)):           
            v = emotions[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fearful':           
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':          
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul
    




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


# model.load_weights(r'C:\Users\jwala\OneDrive\Desktop\Emotion-based-music-recommendation-system-main\model.h5')
model.load_weights("model.h5")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

print("Loading Haarcascade Classifier...")
# face = cv2.CascadeClassifier(r'C:\Users\jwala\OneDrive\Desktop\Emotion-based-music-recommendation-system-main\haarcascade_frontalface_default.xml')
face  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>"
            , unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>"
            , unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

emotion_list = [] 
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION (Click here)'):

        emotion_list.clear()
        cap = cv2.VideoCapture(0)

        frame_box = st.image([])

        for _ in range(20):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.reshape(1, 48, 48, 1)

                prediction = model.predict(roi_gray, verbose=0)
                emotion = emotion_dict[np.argmax(prediction)]
                emotion_list.append(emotion)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame_rgb)


            # if cv2.waitKey(1) & 0xFF == ord('s'):
            #     break
            # if count >= 20:
            #     break
        cap.release()

        if emotion_list:
            emotion_list = pre(emotion_list)
            st.success(f"Emotion detected: {emotion_list}")
        else:
            st.warning("No face detected. Try better lighting.")
 
        

with col3:
    pass

# Generate recommendations only if emotions exist
if emotion_list:
    new_df = fun(emotion_list)
else:
    new_df = pd.DataFrame()

st.write("")

st.markdown(
    "<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>",
    unsafe_allow_html=True
)

st.write("---------------------------------------------------------------------------------------------------------------------")

if not new_df.empty:
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        st.markdown(
            f"<h4 style='text-align: center;'><a href={l}>{i+1} - {n}</a></h4>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>",
            unsafe_allow_html=True
        )
        st.write("---------------------------------------------------------------------------------------------------------------------")
else:
    st.info("No recommendations available. Please scan emotion first.")
