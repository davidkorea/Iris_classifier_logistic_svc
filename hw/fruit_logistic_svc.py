import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DATA_FILE = '../data_ai/fruit_data.csv'
FEAT_COLS = ['mass','width','height','color_score']
LABEL_DICT = {
    'apple':0,
    'mandarin':1,
    'orange':2,
    'lemon':3
}

fruit_data = pd.read_csv(DATA_FILE)
X = fruit_data[FEAT_COLS].values
y = fruit_data['fruit_name'].map(LABEL_DICT).values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=20)

model_dict = {
    'kNN':KNeighborsClassifier(n_neighbors=3),
    'Logistic Reg':LogisticRegression(C=1e2),
    'SVC':SVC(C=1)
}

for model_name, model in model_dict.items():
    model.fit(X_train,y_train)
    accuracy = model.score(X_test,y_test)
    print('Model: {} - Accuracy: {}'.format(model_name,accuracy))


# Results:
# Model: kNN - Accuracy: 0.9166666666666666
# Model: Logistic Reg - Accuracy: 0.8333333333333334
# Model: SVC - Accuracy: 0.75