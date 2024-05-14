import os
import mediapipe as mp
import cv2
import pickle

mp_kädet = mp.solutions.hands
mp_piirto = mp.solutions.drawing_utils
mp_piirto_tyylit = mp.solutions.drawing_styles

kädet = mp_kädet.Hands(static_image_mode=True, min_detection_confidence=0.3)

sijainti = './data'


data = []
labels = []
for kansio in os.listdir(sijainti):
    for img_path in os.listdir(os.path.join(sijainti, kansio)):
        data_aux = []
        kuva = cv2.imread(os.path.join(sijainti, kansio, img_path))
        kuva_rgb = cv2.cvtColor(kuva, cv2.COLOR_BGR2RGB)

        tulokset = kädet.process(kuva_rgb)
        if tulokset.multi_hand_landmarks:
            for hand_landmarks in tulokset.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(kansio)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()