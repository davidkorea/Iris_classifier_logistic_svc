import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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
        'kNN': (KNeighborsClassifier(),
                {
                    'n_neighbors': [10,15,25],
                    'p': [1, 2]
                }),
        'Logistic Reg': (LogisticRegression(),
                         {
                             'C': [1e-2, 1, 1e2]
                         }),
        'SVM': (SVC(),
                {
                    'C': [1e-2, 1, 1e2]
                })
    }

    for model_name, (model, paras) in model_dict.items():
        classifier = GridSearchCV(estimator=model, param_grid=paras, cv=5)
        classifier.fit(X_train,y_train)
        best_clf = classifier.best_estimator_

        acc = best_clf.score(X_test,y_test)
        print('Model:{} - Accuracy:{} - Paras: {}'.format(model_name,acc,classifier.best_params_))

main()

# Results:
# Model:kNN - Accuracy:0.96 - Paras: {'n_neighbors': 15, 'p': 2}
# Model:Logistic Reg - Accuracy:0.96 - Paras: {'C': 100.0}
# Model:SVM - Accuracy:0.98 - Paras: {'C': 1}