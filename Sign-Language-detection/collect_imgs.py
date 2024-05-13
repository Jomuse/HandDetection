import os
import cv2
sijainti = './data' #pathin asetus eli mihin tiedosto menee
if not os.path.exists(sijainti): #tarkistaa onko tiedostoa, jos ei ole luo sen.
    os.makedirs(sijainti)
luokat = 3  #asetetaan käsi merkkien määrä
kuvaloppu = 3 #asettaan haluttu määrä kuvia
kamera = cv2.VideoCapture(0) # asettaa integroidun kuvausta varten
for j in range(luokat):    #Looppaa käsimerkkien verran
    if not os.path.exists(os.path.join(sijainti, str(j))): #luo kansiot käsi merkeille
        os.makedirs(os.path.join(sijainti, str(j)))
    print('kerätään dataa merkille {}'.format(j)) #ilmoittaa mielle merkille otetaan kuvia
    while True: 
        ret, frame = kamera.read() #lukee kameraa
        cv2.putText(frame, 'Oletko valmis? Paina "E" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
        cv2.LINE_AA) #kysyy oletko valmis ruudulla
        cv2.imshow('frame', frame) 
        if cv2.waitKey(25) == ord('e'): #aloittaa kuvauksen
            break
    kuva_alku = 0
    while kuva_alku < kuvaloppu: #ottaa kuvat ja lisää kuvat oikeaan kansioon
        ret, frame = kamera.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(sijainti, str(j), '{}.jpg'.format(kuva_alku)), frame)
        kuva_alku += 1
kamera.release()#kameran käytön
cv2.destroyAllWindows() #lopettaa koodin