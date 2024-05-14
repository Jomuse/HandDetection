import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

malli = RandomForestClassifier()

malli.fit(x_train, y_train)

y_ennuste = malli.predict(x_test)

tarkkuus = accuracy_score(y_ennuste, y_test)

print('{}% näytteistä luokiteltiin oikein !'.format(tarkkuus * 100))

f = open('model.p', 'wb')
pickle.dump({'model': malli}, f)
f.close()