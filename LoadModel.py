import win32com.client
import time
from joblib import dump, load
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

N = 9
fs = 250

model = load("model_RFR.joblib")
print(model)
X = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=0)
X = np.reshape(X, (N, fs))
Y = np.zeros(shape=(N, 1))
Y[0:3] = 1
Y[3:6] = 2
Y[6:9] = 0

#xtest = np.zeros(shape=(1, fs))
Application = win32com.client.Dispatch("PowerPoint.Application")
Presentation = Application.Presentations.Open("D:\\dataset\\slide.pptx")
print(Presentation.Name)
Presentation.SlideShowSettings.Run()
time.sleep(5)

for i in range(N):
    xfft = np.abs(fft(X[i,:])[:fs//2])
    xfft = np.reshape(xfft, (1, fs//2));
    ypred = model.predict(xfft)
    print("Classfiy result: ", ypred)
    if ypred == 1:
        print("\tNEXT SLIDE")
    elif ypred == 2:
        print("\tPREVIOUS SLIDE")    
    elif ypred == 0:
        print("\tNORMAL")  

    if ypred == 1:
        Presentation.SlideShowWindow.View.Next()
        time.sleep(3)
    elif ypred == 2:
        Presentation.SlideShowWindow.View.Previous()
        time.sleep(3)  
    elif ypred == 0:
        time.sleep(3) 
print("End of Data. Close Slide Show")
Application.Quit()
'''
xTrain, xTest, yTrain, yTest = train_test_split( X, Y, stratify=Y,train_size=0.8, test_size=0.2)
print("> Test set accuracy: {:.3f}".format(model.score(xTest, yTest)))
yPred = model.predict(xTest)
precision, recall,  fscore, support = precision_recall_fscore_support(yTest, yPred)
print(precision)
print(recall)
print(fscore)
#plot_confusion_matrix(model, xTest, yTest)
#plt.show()
'''