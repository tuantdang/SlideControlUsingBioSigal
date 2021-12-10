from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from joblib import dump, load
from scipy.fft import fft, fftfreq
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

SVM = 1
KNN = 2
RFR = 3
#
modelSelection = 1

#Parameter
nClasses = 3
nChunk = 25
fs = 250
T = 1/fs


X_raw= np.zeros(shape=(nClasses*nChunk, fs))
X = np.zeros(shape=(nClasses*nChunk, fs//2))
Y = np.zeros(shape=(nClasses*nChunk, 1))

#Eye Blink (EOG)
for i in range(nChunk):
    fileName = "clean_data\\eyeblink" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0)        
    X_raw[i,:] = x
Y[0:25] = 1

#Grindteeth (EMG)
x = np.loadtxt(open("clean_data\\Grindteeth.csv", "rb"), delimiter=",", skiprows=0)    
x = np.reshape(x, (nChunk, fs))
X_raw[25:50,] = x
Y[25:50] = 2

#Normal
x = np.loadtxt(open("clean_data\\normal.csv", "rb"), delimiter=",", skiprows=0)    
x = np.reshape(x, (nChunk, fs))
X_raw[50:75,] = x
Y[50:75] = 0

for i in range(nChunk*nClasses):
    #yf = np.abs(fft(X_raw[i,:])[:fs//2])
    yf = fft(X_raw[i,:])[:fs//2]
    X[i,:] = yf

'''
iSample = 70
xf = fftfreq(fs, T)[:fs//2]
fig, axs = plt.subplots(2)
fig.suptitle('Normal')
axs[0].plot(np.linspace(1,250, 250), X_raw[iSample,:])
axs[1].plot(xf, 2.0/fs * np.abs(X[iSample, 0:fs//2]))
plt.grid()
plt.show()
'''

xTrain, xTest, yTrain, yTest = train_test_split( X, Y, stratify=Y,train_size=0.8, test_size=0.2)
if modelSelection == SVM:
    model = LinearSVC(penalty = 'l2', C = .1380000000000001, tol = .008, max_iter = 1000, dual = False, class_weight = 'balanced')
elif modelSelection == KNN:
    model = KNeighborsClassifier(n_neighbors=1)
elif modelSelection == RFR:
    model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(xTrain, yTrain)
dump(model, "model.joblib")


#print("> Train set accuracy: {:.3f}".format(model.score(xTrain, yTrain)))
acc = model.score(xTest, yTest)
print("> Test set accuracy: {:.3f}".format(acc))
#print("> Entire dataset accuracy: {:.3f}".format(model.score(X, Y)))

yPred = model.predict(xTest)
precision, recall,  fscore, support = precision_recall_fscore_support(yTest, yPred)
print(precision)
print(recall)
print(fscore)

scores = cross_val_score(model, X, Y, cv=5)
for x, acc in enumerate(scores):
    print("Fold " + str(x+1) + ". Accuracy : " + "{:.2f}".format(acc))

plot_confusion_matrix(model, xTest, yTest)
if modelSelection == SVM:
    plt.title("Confusion matrix: SVM. Accuracy = " + "{:.2f}".format((acc)))
elif modelSelection == KNN:
    plt.title("Confusion matrix: KNN. Accuracy = " + "{:.2f}".format((acc)))
elif modelSelection == RFR:    
    plt.title("Confusion matrix: RFR. Accuracy = " + "{:.2f}".format((acc)))
plt.show()


