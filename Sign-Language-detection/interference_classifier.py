import cv2
import mediapipe as mp
import pickle
import numpy as np

malli_kansio = pickle.load(open('./model.p', 'rb')) #ladataan malli pickle-tiedostosta
malli = malli_kansio['model']
cap = cv2.VideoCapture(0) #kameran asetus
mp_kädet = mp.solutions.hands #MediaPipen käsien tunnistuksen ja piirto-ominaisuuksien avustaminen
mp_piirto = mp.solutions.drawing_utils
mp_piirto_tyylit = mp.solutions.drawing_styles

kädet = mp_kädet.Hands(static_image_mode=True, min_detection_confidence=0.3) #käsien tunnistuksen alustaminen
labels_dict = {0: 'kivi', 1: 'paperi', 2: 'sakset'} #asettaa nimet datalle
while True: #videon lukeminen ja käsittely aloitetaan
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read() #luetaan nykyinen frame videosta
    H, W, _ = frame.shape #framen mitat
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #framen muuttaminen rgb-muotoon mediapipea varten
    tulokset = kädet.process(frame_rgb) #prosessoidaan frame käsien landmarkien tunnistamiseksi
    if tulokset.multi_hand_landmarks:
        for hand_landmarks in tulokset.multi_hand_landmarks: #piirretään landmarkit ja yhteydet
            mp_piirto.draw_landmarks(
                frame,
                hand_landmarks,
                mp_kädet.HAND_CONNECTIONS,
                mp_piirto_tyylit.get_default_hand_landmarks_style(),
                mp_piirto_tyylit.get_default_hand_connections_style()
            )
        for hand_landmarks in tulokset.multi_hand_landmarks: #käydään jokainen käsi videosta ja kerätään landmarkien koordinaatit
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        x1 = int(min(x_) * W) - 10 #ääriviivojen koordinaatit
        y1 = int(min(y_) * H) -10 
        x2 = int(max(x_) * W) -10
        y2 = int(max(y_) * H) -10
        ennuste = malli.predict([np.asarray(data_aux)]) #käsimerkin ennustaminen
        ennuste_character = labels_dict[int(ennuste[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) #suorakulmion piirto käden ympärille
        cv2.putText(frame, ennuste_character, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)
    cv2.imshow('frame', frame) #nykyisen framen näyttö
    cv2.waitKey(1) #odotetaan 1 ms ennen seuraavan framen lukemista

cap.release()
cv2.destroyAllWindows()