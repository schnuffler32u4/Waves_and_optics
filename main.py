import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import warnings

# AI model start here

fsttrain = pd.read_csv('Calibration_data/Calibration1stpend_01_12_training.csv')
fsttest = pd.read_csv('Calibration_data/Calibration1stpend_01_12_testing.csv')
sndtrain = pd.read_csv('Calibration_data/Calibration2ndpend_01_12_training.csv')
sndtest = pd.read_csv('Calibration_data/Calibration2ndpend_01_12_testing.csv')

# We give the number 0 to the first pendulum and the number 1 to the second pendulum

id0train = np.zeros(len(fsttrain), dtype=int)
id0test = np.zeros(len(fsttest), dtype=int)
id1train = np.ones(len(sndtrain), dtype=int)
id1test = np.ones(len(sndtest), dtype=int)

fsttrain["Id"] = id0train
fsttest["Id"] = id0test
sndtrain["Id"] = id1train
sndtest["Id"] = id1test

traindata = pd.concat((fsttrain, sndtrain))
testdata = pd.concat((fsttest, sndtest))

traindata.rename(columns={"Time (s) Run #1": "Time", "Light Intensity (lx) Run #1": "Light"}, inplace=True)
testdata.rename(columns={"Time (s) Run #1": "Time", "Light Intensity (lx) Run #1": "Light"}, inplace=True)

traindata[["Time", "Light", "Id"]].describe()

colors = ['#e09028', '#00FFFF']

# _ = pd.plotting.scatter_matrix(traindata, figsize=(14,14), diagonal='kde', alpha=0.6, color=[colors[x-1] for x in list(traindata.Id)])

# plt.show()

# Remove unneeded datapoints

traindata.drop(traindata[traindata.Light > 260].index, inplace=True)
testdata.drop(testdata[testdata.Light > 260].index, inplace=True)

# traindata.reset_index(inplace=True)
# testdata.reset_index(inplace=True)

traindata.index = pd.RangeIndex(len(traindata.index))
testdata.index = pd.RangeIndex(len(testdata.index))

print(traindata)
# print(type(traindata.iloc[-1]['T']))

# Normalize the features of the data to make it easier for the classifier

traindata["T_normal"] = (traindata.Time - traindata.Time.mean()) / traindata.Time.std()
traindata["Light_normal"] = (traindata.Light - traindata.Light.mean()) / traindata.Light.std()
testdata["T_normal"] = (testdata.Time - testdata.Time.mean()) / testdata.Time.std()
testdata["Light_normal"] = (testdata.Light - testdata.Light.mean()) / testdata.Light.std()


def plot_decision_surface(clf, data, feature_list, step=0.02):
    """
    Function that creates a decision surface of classifier "clf"
    together with features from the two entries of the "features_list" of "data"
    """
    X = data[feature_list[0]].values
    Y = data[feature_list[1]].values

    # colors associated with the three classes, light is used for surface and bold for datapoints
    cmap_light = ListedColormap(['#e09028', '#00FFFF'])
    cmap_bold = ListedColormap(['#59380d', '#008080'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max] x [y_min, y_max].
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Put the result into a color plot
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    # Plot the data points
    plt.scatter(X, Y, s=30, c=data['Id'], cmap=cmap_bold)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(feature_list[0])
    plt.ylabel(feature_list[1])
    plt.show()
    return


def kNN_classify(traindata, testdata,
                 k=15,
                 prop_test=0,
                 feature_list=['T_normal', 'Light_normal'],
                 plot_ds=False):
    """
    Use kNN algorithm to classify dataset data. Use feature list feature_list
    and use prop_test fraction of the data as test sample, rest for training.
    The function returns an updated dataset including a column with predicted class.
    """

    # assert 0 < prop_test < 1

    #
    # make a copy of the dataframe. The original table will not be altered.
    #
    # data_tmp = data.copy()
    traindata_tmp = traindata.copy()
    testdata_tmp = testdata.copy()
    #
    # Add a column specifying whether event is used for training or for testing
    #
    traindata_tmp['is_train'] = np.ones(len(traindata_tmp))
    testdata_tmp['is_train'] = np.zeros((len(testdata_tmp)))
    data_tmp = pd.concat((testdata_tmp, traindata_tmp))
    #
    # Extract two seperate datasets for training (train) and testing (test)
    #
    train, test = data_tmp[data_tmp['is_train'] == True], data_tmp[data_tmp['is_train'] == False]

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(train[feature_list], train['Id'])

    # evaluate the training points
    trainpred = clf.predict(train[feature_list])
    # train['prediction'] = [x for x in trainpred]
    train['prediction'] = trainpred
    # Predict on the testing points
    testpred = clf.predict(test[feature_list])
    # test['prediction'] = [x for x in testpred]
    test['prediction'] = testpred
    # data_update = train.append(test)
    data_update = pd.concat((train, test))
    # plotting the decision surface for the first two given features

    if (plot_ds) & (len(feature_list) == 2):
        plot_decision_surface(clf, data_update, feature_list)

    return data_update


def getFOM(data):
    # define S and B
    nrtrain_S = len(data[(data.is_train == True) & (data.prediction == data.Id) & (data.Id == 0)])
    nrtrain_B = len(data[(data.is_train == True) & (data.prediction != data.Id) & (data.Id == 1)])
    nrtest_S = len(data[(data.is_train == False) & (data.prediction == data.Id) & (data.Id == 0)])
    nrtest_B = len(data[(data.is_train == False) & (data.prediction != data.Id) & (data.Id == 1)])

    return nrtest_S / np.sqrt(nrtest_S + nrtest_B)


things = np.zeros(100)

for k in range(1, len(things) + 1):
    datanew = kNN_classify(traindata, testdata, k=k)
    things[k - 1] = float(getFOM(datanew))
k = int(np.where(things == np.amax(things))[0] + 1)
datanew = kNN_classify(traindata, testdata, plot_ds=True, k=k)
fomtest = getFOM(datanew)
print("The FOM is:")
print(fomtest)
print("And the correct identification rate is: " + str(len(datanew[(datanew.is_train == False) & (datanew.prediction == datanew.Id) & (datanew.Id == 0)])) + " out of " +str(len(testdata[(testdata.Id == 0)])))

