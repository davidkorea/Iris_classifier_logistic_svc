import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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
def main():
    fruit_data = pd.read_csv(DATA_FILE)
    X = fruit_data[FEAT_COLS].values
    y = fruit_data['fruit_name'].map(LABEL_DICT).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=20)

    model_dict = {
        'kNN': (KNeighborsClassifier(),
                {
                    'n_neighbors': [3,5,10],
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
        classifier = GridSearchCV(estimator=model, param_grid=paras, cv=3)
        # if cv is too big, WARNING:
        # Warning: The least populated class in y has only 3 members,
        # which is too few. The minimum number of members in any class
        # cannot be less than n_splits=5.% (min_groups, self.n_splits)),
        # Warning)
        classifier.fit(X_train,y_train)
        best_clf = classifier.best_estimator_

        acc = best_clf.score(X_test,y_test)
        print('Model:{} - Accuracy:{:.2f} - Paras: {}'.format(model_name,acc,classifier.best_params_))

main()

# Results:
# Model:kNN - Accuracy:0.92 - Paras: {'n_neighbors': 3, 'p': 1}
# Model:Logistic Reg - Accuracy:0.83 - Paras: {'C': 100.0}
# Model:SVM - Accuracy:0.75 - Paras: {'C': 1}