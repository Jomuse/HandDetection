import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb')) #avataan pickle-tiedosto
data = np.asarray(data_dict['data']) #numpyn avulla asetetaan datan arvot listaan
labels = np.asarray(data_dict['labels']) #numpyn avulla asetetaan labels arvot listaan
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) #määritellään testien ja treenaus datojen määrät. Tässä testit ovat 20% määrästä, shuffle sekoittaa kuvat
malli = RandomForestClassifier() #Satunnaismetsämallin asetus
malli.fit(x_train, y_train) #asetetaan treenausdata malliin
y_ennuste = malli.predict(x_test) #mallin ennuste
tarkkuus = accuracy_score(y_ennuste, y_test) #tarkuuden määrä
print('{}% näytteistä luokiteltiin oikein !'.format(tarkkuus * 100)) #tulostetaan mallin luokittelun määrä prosentteina.
f = open('model.p', 'wb') #asetetaan mallin tulokset pickle-tiedostoon
pickle.dump({'model': malli}, f)
f.close()