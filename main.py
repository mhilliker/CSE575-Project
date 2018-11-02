import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import sys
import operator
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers.normalization import BatchNormalization

# features that need to be turned into discrete values
features_to_categorize = ["OSOURCE", "STATE", "ZIP", "MAILCODE", "PVASTATE", "NOEXCH", "RECINHSE", "RECP3",
                              "RECPGVG",
                              "RECSWEEP", "MDMAUD", "DOMAIN", "CLUSTER", "AGEFLAG", "HOMEOWNR", "CHILD03", "CHILD07",
                              "CHILD12",
                              "CHILD18", "GENDER", "DATASRCE", "SOLP3", "SOLIH", "MAJOR", "WEALTH2", "GEOCODE",
                              "COLLECT1",
                              "VETERANS", "BIBLE", "CATLG", "HOMEE", "PETS", "CDPLAY", "STEREO", "PCOWNERS", "PHOTO",
                              "CRAFTS",
                              "FISHER", "GARDENIN", "BOATS", "WALKER", "KIDSTUFF", "CARDS", "LIFESRC", "PEPSTRFL",
                              "CLUSTER2",
                              "GEOCODE2", "RFA_2R", "RFA_2F", "RFA_2A", "MDMAUD_R", "MDMAUD_F", "MDMAUD_A", "INCOME",
                              "AGE",
                              "WEALTH1", "NUMCHLD", "TIMELAG", "PUBHLTH", "MBCRAFT", "MBGARDEN", "RAMNT_24", "NEXTDATE",
                              "DMA", "ADI", "MSA", "ADATE_24", "PUBOPP", "PUBPHOTO", "MBBOOKS", "MAGFAML", "MAGFEM",
                              "PUBNEWFN", "PUBGARDN", "PUBCULIN", "PUBDOITY", "MAGMALE", "MBCOLECT", "RDATE_24",
                              "PLATES",
                              "RFA_24"]

# all features except for TARGET_B
all_features = ["ODATEDW", "OSOURCE", "TCODE", "STATE", "ZIP", "MAILCODE", "PVASTATE", "DOB", "NOEXCH", "RECINHSE",
                "RECP3", "RECPGVG", "RECSWEEP", "MDMAUD", "DOMAIN", "CLUSTER", "AGE", "AGEFLAG", "HOMEOWNR",
                "CHILD03", "CHILD07", "CHILD12", "CHILD18", "NUMCHLD", "INCOME", "GENDER", "WEALTH1", "HIT",
                "MBCRAFT", "MBGARDEN", "MBBOOKS", "MBCOLECT", "MAGFAML", "MAGFEM", "MAGMALE", "PUBGARDN",
                "PUBCULIN", "PUBHLTH", "PUBDOITY", "PUBNEWFN", "PUBPHOTO", "PUBOPP", "DATASRCE", "MALEMILI",
                "MALEVET", "VIETVETS", "WWIIVETS", "LOCALGOV", "STATEGOV", "FEDGOV", "SOLP3", "SOLIH", "MAJOR",
                "WEALTH2", "GEOCODE", "COLLECT1", "VETERANS", "BIBLE", "CATLG", "HOMEE", "PETS", "CDPLAY", "STEREO",
                "PCOWNERS", "PHOTO", "CRAFTS", "FISHER", "GARDENIN", "BOATS", "WALKER", "KIDSTUFF", "CARDS",
                "PLATES", "LIFESRC", "PEPSTRFL", "POP901", "POP902", "POP903", "POP90C1", "POP90C2", "POP90C3",
                "POP90C4", "POP90C5", "ETH1", "ETH2", "ETH3", "ETH4", "ETH5", "ETH6", "ETH7", "ETH8", "ETH9",
                "ETH10", "ETH11", "ETH12", "ETH13", "ETH14", "ETH15", "ETH16", "AGE901", "AGE902", "AGE903",
                "AGE904", "AGE905", "AGE906", "AGE907", "CHIL1", "CHIL2", "CHIL3", "AGEC1", "AGEC2", "AGEC3",
                "AGEC4", "AGEC5", "AGEC6", "AGEC7", "CHILC1", "CHILC2", "CHILC3", "CHILC4", "CHILC5", "HHAGE1",
                "HHAGE2", "HHAGE3", "HHN1", "HHN2", "HHN3", "HHN4", "HHN5", "HHN6", "MARR1", "MARR2", "MARR3",
                "MARR4", "HHP1", "HHP2", "DW1", "DW2", "DW3", "DW4", "DW5", "DW6", "DW7", "DW8", "DW9", "HV1",
                "HV2", "HV3", "HV4", "HU1", "HU2", "HU3", "HU4", "HU5", "HHD1", "HHD2", "HHD3", "HHD4", "HHD5",
                "HHD6", "HHD7", "HHD8", "HHD9", "HHD10", "HHD11", "HHD12", "ETHC1", "ETHC2", "ETHC3", "ETHC4",
                "ETHC5", "ETHC6", "HVP1", "HVP2", "HVP3", "HVP4", "HVP5", "HVP6", "HUR1", "HUR2", "RHP1", "RHP2",
                "RHP3", "RHP4", "HUPA1", "HUPA2", "HUPA3", "HUPA4", "HUPA5", "HUPA6", "HUPA7", "RP1", "RP2", "RP3",
                "RP4", "MSA", "ADI", "DMA", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6", "IC7", "IC8", "IC9", "IC10",
                "IC11", "IC12", "IC13", "IC14", "IC15", "IC16", "IC17", "IC18", "IC19", "IC20", "IC21", "IC22",
                "IC23", "HHAS1", "HHAS2", "HHAS3", "HHAS4", "MC1", "MC2", "MC3", "TPE1", "TPE2", "TPE3", "TPE4",
                "TPE5", "TPE6", "TPE7", "TPE8", "TPE9", "PEC1", "PEC2", "TPE10", "TPE11", "TPE12", "TPE13", "LFC1",
                "LFC2", "LFC3", "LFC4", "LFC5", "LFC6", "LFC7", "LFC8", "LFC9", "LFC10", "OCC1", "OCC2", "OCC3",
                "OCC4", "OCC5", "OCC6", "OCC7", "OCC8", "OCC9", "OCC10", "OCC11", "OCC12", "OCC13", "EIC1", "EIC2",
                "EIC3", "EIC4", "EIC5", "EIC6", "EIC7", "EIC8", "EIC9", "EIC10", "EIC11", "EIC12", "EIC13", "EIC14",
                "EIC15", "EIC16", "OEDC1", "OEDC2", "OEDC3", "OEDC4", "OEDC5", "OEDC6", "OEDC7", "EC1", "EC2",
                "EC3", "EC4", "EC5", "EC6", "EC7", "EC8", "SEC1", "SEC2", "SEC3", "SEC4", "SEC5", "AFC1", "AFC2",
                "AFC3", "AFC4", "AFC5", "AFC6", "VC1", "VC2", "VC3", "VC4", "ANC1", "ANC2", "ANC3", "ANC4", "ANC5",
                "ANC6", "ANC7", "ANC8", "ANC9", "ANC10", "ANC11", "ANC12", "ANC13", "ANC14", "ANC15", "POBC1",
                "POBC2", "LSC1", "LSC2", "LSC3", "LSC4", "VOC1", "VOC2", "VOC3", "HC1", "HC2", "HC3", "HC4", "HC5",
                "HC6", "HC7", "HC8", "HC9", "HC10", "HC11", "HC12", "HC13", "HC14", "HC15", "HC16", "HC17", "HC18",
                "HC19", "HC20", "HC21", "MHUC1", "MHUC2", "AC1", "AC2", "ADATE_2", "ADATE_3", "ADATE_4", "ADATE_5",
                "ADATE_6", "ADATE_7", "ADATE_8", "ADATE_9", "ADATE_10", "ADATE_11", "ADATE_12", "ADATE_13",
                "ADATE_14", "ADATE_15", "ADATE_16", "ADATE_17", "ADATE_18", "ADATE_19", "ADATE_20", "ADATE_21",
                "ADATE_22", "ADATE_23", "ADATE_24", "RFA_2", "RFA_3", "RFA_4", "RFA_5", "RFA_6", "RFA_7", "RFA_8",
                "RFA_9", "RFA_10", "RFA_11", "RFA_12", "RFA_13", "RFA_14", "RFA_15", "RFA_16", "RFA_17", "RFA_18",
                "RFA_19", "RFA_20", "RFA_21", "RFA_22", "RFA_23", "RFA_24", "CARDPROM", "MAXADATE", "NUMPROM",
                "CARDPM12", "NUMPRM12", "RDATE_3", "RDATE_4", "RDATE_5", "RDATE_6", "RDATE_7", "RDATE_8", "RDATE_9",
                "RDATE_10", "RDATE_11", "RDATE_12", "RDATE_13", "RDATE_14", "RDATE_15", "RDATE_16", "RDATE_17",
                "RDATE_18", "RDATE_19", "RDATE_20", "RDATE_21", "RDATE_22", "RDATE_23", "RDATE_24", "RAMNT_3",
                "RAMNT_4", "RAMNT_5", "RAMNT_6", "RAMNT_7", "RAMNT_8", "RAMNT_9", "RAMNT_10", "RAMNT_11",
                "RAMNT_12", "RAMNT_13", "RAMNT_14", "RAMNT_15", "RAMNT_16", "RAMNT_17", "RAMNT_18", "RAMNT_19",
                "RAMNT_20", "RAMNT_21", "RAMNT_22", "RAMNT_23", "RAMNT_24", "RAMNTALL", "NGIFTALL", "CARDGIFT",
                "MINRAMNT", "MINRDATE", "MAXRAMNT", "MAXRDATE", "LASTGIFT", "LASTDATE", "FISTDATE", "NEXTDATE",
                "TIMELAG", "AVGGIFT", "CONTROLN", "HPHONE_D", "RFA_2R", "RFA_2F", "RFA_2A", "MDMAUD_R", "MDMAUD_F",
                "MDMAUD_A", "CLUSTER2", "GEOCODE2"]


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def show_pca(df, num_components=10):
    print("PCA with ", num_components, " components")
    data_scaled = pd.DataFrame(scale(df), columns=df.columns)
    pca = PCA(n_components=num_components)
    pca.fit_transform(data_scaled)
    print(pd.DataFrame(pca.components_, columns=data_scaled.columns, index=["PCA-{0}".format(x) for x in range(num_components)]))


def label_encode(df, features):
    lb_make = LabelEncoder()
    for feature in features:
        try:
            df[feature] = lb_make.fit_transform(df[feature])
        except:
            print("failed to encode {0}".format(feature))


def clean_data(data, is_training_set=False):
    for i in range(2, 24):
        features_to_categorize.append("ADATE_{0}".format(i))
        features_to_categorize.append("RFA_{0}".format(i))
    for i in range(3, 24):
        features_to_categorize.append("RDATE_{0}".format(i))
        features_to_categorize.append("RAMNT_{0}".format(i))

    # these mix string, null, and int so it freaks out
    data[['NOEXCH', 'GEOCODE2']] = data[['NOEXCH', 'GEOCODE2']].astype(str)
    label_encode(data, features_to_categorize)
    print("categorization complete")

    used_features = all_features
    if is_training_set:
        data = data[used_features + ["TARGET_B"]]
    else:
        data = data[used_features]
    # Cleaning data set of NaN
    num_rows_before_clean = len(data)
    data = data.dropna(axis=0, how='any')
    num_rows_after_clean = len(data)
    # those numbers should be the same if all of the data has been properly cleaned
    if num_rows_before_clean != num_rows_after_clean:
        print("Lost {0} rows when cleaning {1} set".format(num_rows_after_clean - num_rows_before_clean,
                                                           "training" if is_training_set else "testing"))

    return data


def report_predictions(X_test, y_pred):
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
        .format(
        X_test.shape[0],
        (X_test["TARGET_B"] != y_pred).sum(),
        100 * (1 - (X_test["TARGET_B"] != y_pred).sum() / X_test.shape[0])
    ))
    print(confusion_matrix(X_test["TARGET_B"], y_pred))
    print(classification_report(X_test["TARGET_B"], y_pred))


def generate_labels(LRN, LRN_pred, output_file):
    pos_count = 0
    with open("predictions.txt", "w") as output_file:
        for index, row in LRN.iterrows():
            control_n = int(row['CONTROLN'])
            label = LRN_pred[index]
            if str(label) != "0":
                pos_count += 1
            val = "{0},{1}\n".format(control_n, label)
            output_file.write(val)
    print("Output file generated successfully. Num positive: " + str(pos_count))


# Importing data set
data = pd.read_csv("cup98LRN.txt")

# Convert categorical variable to numeric
data = clean_data(data, True)


# Split data set in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.8, random_state=int(time.time()))


# check PCA of Data
show_pca(X_train, 2)


# Train classifiers

used_features = all_features # in the future, select features more intelligently

#LRN = pd.read_csv("cup98VAL.txt")
#LRN = clean_data(LRN)

def bernoulli_naive_bayes(X_train, X_test, used_features, validation_data=None):
    print("BernoulliNB:")
    bnb = BernoulliNB()
    bnb.fit(
        X_train[used_features].values,
        X_train["TARGET_B"]
    )
    y_pred = bnb.predict(X_test[used_features])
    report_predictions(X_test, y_pred)

    if validation_data is not None:
        validation_prediction = bnb.predict(validation_data)
        generate_labels(validation_data, validation_prediction, "BNBlabels.csv")

bernoulli_naive_bayes(X_train, X_test, used_features)


@timeit
def logistic_regression(X_train, X_test, used_features, validation_data=None):
    print("Logistic Regression:")
    log_reg = LogisticRegression(solver='sag', max_iter=300, C=.01)
    log_reg.fit(X_train[used_features].values,
               X_train["TARGET_B"])
    y_pred = log_reg.predict(X_test[used_features])
    report_predictions(X_test, y_pred)

    if validation_data is not None:
        validation_prediction = log_reg.predict(validation_data)
        generate_labels(validation_data, validation_prediction, "LRlabels.csv")


logistic_regression(X_train, X_test, used_features)


def decision_tree_classifier(X_train, X_test, used_features, validation_data=None, show_feature_importance=False):
    print("Decision Trees:")
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train[used_features].values,
                   X_train["TARGET_B"])
    y_pred = classifier.predict(X_test[used_features])
    report_predictions(X_test, y_pred)

    if validation_data is not None:
        validation_prediction = classifier.predict(validation_data)
        generate_labels(validation_data, validation_prediction, "DTClabels.csv")

    if show_feature_importance is False:
        return

    print("Top Features:")
    feature_ranks = []
    for i in range(len(classifier.feature_importances_)):
        feature_ranks.append((classifier.feature_importances_[i], used_features[i]))
    feature_ranks.sort(key=operator.itemgetter(0), reverse=True)
    for i in range(min(15, len(feature_ranks))):
        importance = "{:10.4f}".format(feature_ranks[i][0])
        print("{0}\t{1}".format(importance, feature_ranks[i][1]))

    pd.Series(classifier.feature_importances_, index=X_train[used_features].columns).plot.bar(color='steelblue',
                                                                                              figsize=(12, 6))

decision_tree_classifier(X_train, X_test, used_features, show_feature_importance=True)


def _feed_forward_nn(X, Y, verbose=True) -> Sequential:
    model = Sequential()
    # model.add(Dense(100, activation='relu', input_dim=479))
    model.add(Dense(100, input_dim=479))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='nadam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, Y, verbose=verbose, shuffle=True, epochs=10, batch_size=16, validation_split=0.2)
    return model


def nn_classifier(X_test, used_features, validation_data=None):
    print("Feed forward neural net (100,100,2)")
    neural_net = _feed_forward_nn(data[used_features].values, data[["TARGET_B"]], verbose=False)
    y_pred = neural_net.predict_classes(X_test[used_features])
    y_list = y_pred.tolist()
    report_predictions(X_test, y_pred)

    if validation_data is not None:
        validation_prediction = neural_net.predict(validation_data)
        generate_labels(validation_data, validation_prediction, "NNlabels.csv")

nn_classifier(X_test, used_features)

"""
NOTES:  
- ZIP needs cleaning before encoding
- MDMAUD should be broken up
- DOMAIN should be broken up
- CHILD* could hot-encode M/F child existance or count
- 
"""



