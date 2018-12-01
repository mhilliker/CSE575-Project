"""This module contains random utilities that helped in the investigation."""

import main
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot


def check_for_bad_data(df):
    """Iterates through frame and finds any bad elements."""
    for row_index, row in df.iterrows():
        for i, el in row.items():
            if not isinstance(el, (int, float, complex)):
                print(i, " - ", str(el))


def decision_tree_features_miner(num_iterations=100, num_features=40):
    """Uses decision trees to extract and rank the best features from the data set."""
    data = main.load_data_from_file(is_training_set=True)
    importance_tracker = {k: 0.0 for v, k in enumerate(main.all_features)}

    for i in range(num_iterations):
        classifier = RandomForestClassifier()
        classifier.fit(data[main.all_features].values, data["TARGET_B"])
        feature_importances = classifier.feature_importances_
        feature_index = 0
        for feature in data.columns:
            if feature == 'TARGET_B':
                continue
            importance_tracker[feature] += feature_importances[feature_index]
            feature_index += 1

    ranked_features = sorted(importance_tracker.items(), key=lambda x: x[1], reverse=True)
    output = []
    for i in range(num_features):
        importance = "{:.6f}".format(ranked_features[i][1])
        print("{0}:\t{1}".format(importance, ranked_features[i][0]))
        output.append(ranked_features[i][0])

    print(str(output))


def tree_tuner():
    """Use cross validation search to find ideal hyperparameters for random forest classifier."""
    data = main.load_data_from_file(is_training_set=True)
    data = data.sample(50000)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=5000, num=5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 12]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    rf = RandomForestClassifier(random_state=42)

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=10, scoring='neg_mean_absolute_error',
                                   cv=3, verbose=2, random_state=42, n_jobs=-1,
                                   return_train_score=True)
    # Fit the random search model
    rf_random.fit(data[main.all_features].values, data["TARGET_B"])

    print("best params:\n" + str(rf_random.best_params_))
    print("\n\n\n\n\n")
    print(str(rf_random.cv_results_))


def voting():
    """Try out voting classifiers to see if this improves performance."""

    data = main.load_data_from_file(is_training_set=True)
    data = main.undersample_data(data, 0, 0.95)
    #data = data.sample(1000)
    used_features = ['RAMNT_11', 'ZIP', 'CONTROLN', 'AVGGIFT', 'TIMELAG', 'OSOURCE', 'MAXRDATE', 'AGE', 'RDATE_11', 'MINRDATE', 'WEALTH1', 'MAGFAML', 'POP903', 'NEXTDATE', 'ADATE_15', 'TPE13', 'ADATE_13', 'WEALTH2', 'RAMNT_8', 'RAMNT_23', 'INCOME', 'ADATE_21', 'ADI', 'LASTDATE', 'POP902', 'RDATE_13', 'RAMNT_14', 'NUMPROM', 'NGIFTALL', 'RAMNTALL', 'RAMNT_13', 'ADATE_23', 'PEC2', 'POP901', 'RAMNT_9', 'EIC4', 'RAMNT_16', 'RDATE_9', 'NUMPRM12', 'VC2']
    X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))
    clf1 = BernoulliNB()
    clf2 = RandomForestClassifier()
    clf3 = DecisionTreeClassifier()
    clf4 = LogisticRegression(solver='sag', max_iter=3000, C=.01)

    eclf1 = VotingClassifier(estimators=[
        ('bnb', clf1),
        ('rf', clf2),
        ('dt', clf3),
        ('lr', clf4),
    ],
    voting='hard', n_jobs=-1)

    eclf1 = eclf1.fit(X_train[used_features].values, X_train["TARGET_B"])
    y_pred = eclf1.predict(X_test[used_features])
    precision = main.report_predictions(X_test, y_pred)

    VAL = main.load_data_from_file(is_training_set=False)
    validation_prediction = eclf1.predict(VAL[used_features])
    main.generate_labels(VAL, validation_prediction, "VBNBlabels.csv")


def simple_svc():
    data = main.load_data_from_file(is_training_set=True)
    data = main.undersample_data(data, 0, 0.75)
    # data = data.sample(1000)
    used_features = ['RAMNT_11', 'ZIP', 'CONTROLN', 'AVGGIFT', 'TIMELAG', 'OSOURCE', 'MAXRDATE', 'AGE', 'RDATE_11',
                     'MINRDATE', 'WEALTH1', 'MAGFAML', 'POP903', 'NEXTDATE', 'ADATE_15', 'TPE13', 'ADATE_13', 'WEALTH2',
                     'RAMNT_8', 'RAMNT_23', 'INCOME', 'ADATE_21', 'ADI', 'LASTDATE', 'POP902', 'RDATE_13', 'RAMNT_14',
                     'NUMPROM', 'NGIFTALL', 'RAMNTALL', 'RAMNT_13', 'ADATE_23', 'PEC2', 'POP901', 'RAMNT_9', 'EIC4',
                     'RAMNT_16', 'RDATE_9', 'NUMPRM12', 'VC2']
    X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()), stratify=data["TARGET_B"])
    classifier = RandomForestClassifier()
    classifier.fit(X_train[used_features].values, X_train["TARGET_B"])
    y_pred = classifier.predict(X_test[used_features])
    main.report_predictions(X_test, y_pred)

    VAL = main.load_data_from_file(is_training_set=False)
    validation_prediction = classifier.predict(VAL[used_features])
    main.generate_labels(VAL, validation_prediction, "UTILlabels.csv")


def get_correlations(data=None, features=None):
    """Get feature correlations."""
    if data is None:
        data = main.load_data_from_file(is_training_set=True)
    if features is None:
        features = main.features_test
    if 'TARGET_B' not in features:
        features += ['TARGET_B']
    data = data[features]
    correlations = data.assign(TARGET_B=data.TARGET_B.astype('category').cat.codes).corr()
    return correlations


def xgboost():
    """Try out xgboost decision tree model."""
    data = main.load_data_from_file(is_training_set=True)
    used_features = main.all_features
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=int(time.time()), stratify=data["TARGET_B"])
    classifier = XGBClassifier(learning_rate=0.1)
    classifier.fit(X_train[used_features].values, X_train["TARGET_B"])
    y_pred = classifier.predict(X_test[used_features].values)
    main.report_predictions(X_test, y_pred)

    plot_importance(classifier)
    pyplot.show()

    VAL = main.load_data_from_file(is_training_set=False)
    validation_prediction = classifier.predict(VAL[used_features].values)
    main.generate_labels(VAL, validation_prediction, "UTILlabels.csv")

def plot_corr(corr):
    fig, ax = plt.subplots(figsize=(16, 16))
    cor_ax = ax.matshow(corr)
    fig.colorbar(cor_ax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

# uncomment one of the below to try them out
# tree_tuner()
# voting()
# simple_svc()
# decision_tree_features_miner()
# corr = get_correlations(features=main.features_100)
# plot_corr(get_correlations(features=main.features_test))
