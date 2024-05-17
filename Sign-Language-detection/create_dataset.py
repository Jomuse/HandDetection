import os
import mediapipe as mp
import cv2
import pickle

#MediaPipen käsien tunnistuksen ja piirto-ominaisuuksien avustaminen
mp_kädet = mp.solutions.hands
mp_piirto = mp.solutions.drawing_utils
mp_piirto_tyylit = mp.solutions.drawing_styles
kädet = mp_kädet.Hands(static_image_mode=True, min_detection_confidence=0.3) #käsien tunnistuksen alustaminen
sijainti = './data' #kuvakansion sijainti
data = []
labels = []
for kansio in os.listdir(sijainti): #käy kansion tiedostoja 
    for img_path in os.listdir(os.path.join(sijainti, kansio)):
        data_aux = [] #apulista käsien koordinaateille
        kuva = cv2.imread(os.path.join(sijainti, kansio, img_path))
        kuva_rgb = cv2.cvtColor(kuva, cv2.COLOR_BGR2RGB) #muutetaan kuvat rgb-muotoon mediapipelle
        tulokset = kädet.process(kuva_rgb) #prosessoi kädet kuvasta landmarkien löytämiseksi
        if tulokset.multi_hand_landmarks:
            for hand_landmarks in tulokset.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)): #landmarkien lisäys listaan
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux) #lisätään koordinaatit data-listaan
            labels.append(kansio)
#tietojen talletus pickle-tiedostoon
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()