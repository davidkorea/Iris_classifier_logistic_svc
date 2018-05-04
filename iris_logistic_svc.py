import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

DATA_FILE = './data_ai/Iris.csv'
# print(os.path.exists(DATA_FILE))
FEATURE_COL = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
SPECIES_LABEL_DICT = {
    'Iris-setosa': 0,  # 山鸢尾
    'Iris-versicolor': 1,  # 变色鸢尾
    'Iris-virginica': 2  # 维吉尼亚鸢尾
}

def main():
    iris_data = pd.read_csv(DATA_FILE)
    X = iris_data[FEATURE_COL].values
    y = iris_data['Species'].map(SPECIES_LABEL_DICT).values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=10)

    model_dict = {
        'kNN':KNeighborsClassifier(n_neighbors=10),
        # 'linear_reg':LinearRegression(),
        # linear_reg is not a classification model, R2 the bigger the better
        'logistic_reg':LogisticRegression(C=1e3),
        'SVC':SVC(C=1e3)
    }

    for model_name, model in model_dict.items():
        model.fit(X_train,y_train)
        accuracy = model.score(X_test,y_test)
        print('Model:{} - Accuracy:{}'.format(model_name,accuracy))

main()

# Results:
# Model:kNN - Accuracy:1.0
# Model:linear_reg - R2:0.9063358327734319 
# Model:logistic_reg - Accuracy:0.98
# Model:SVC - Accuracy:0.92