import cv2
import mediapipe as mp
import pickle
import numpy as np

malli_kansio = pickle.load(open('./model.p', 'rb'))
malli = malli_kansio['model']

cap = cv2.VideoCapture(0)

mp_kädet = mp.solutions.hands
mp_piirto = mp.solutions.drawing_utils
mp_piirto_tyylit = mp.solutions.drawing_styles

kädet = mp_kädet.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'kivi', 1: 'paperi', 2: 'sakset'}
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tulokset = kädet.process(frame_rgb)
    if tulokset.multi_hand_landmarks:
        for hand_landmarks in tulokset.multi_hand_landmarks:
            mp_piirto.draw_landmarks(
                frame,
                hand_landmarks,
                mp_kädet.HAND_CONNECTIONS,
                mp_piirto_tyylit.get_default_hand_landmarks_style(),
                mp_piirto_tyylit.get_default_hand_connections_style()
            )
        for hand_landmarks in tulokset.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) -10 

        x2 = int(max(x_) * W) -10
        y2 = int(max(y_) * H) -10

        ennuste = malli.predict([np.asarray(data_aux)])

        ennuste_character = labels_dict[int(ennuste[0])]
        


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, ennuste_character, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                            cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()