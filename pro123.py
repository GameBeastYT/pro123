import cv2
import numpy as np
import pandas as pd
#import seaborn as sb
#import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps as io
import os, ssl, time

if(not os.environ.get("PYTHONHTTPSVERIFY", "")and getattr(ssl, "_create_verified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n = len(classes)

xtrain, xtest, ytrain, ytest = tts(x, y, random_state = 0, train_size = 10725, test_size = 3575)
xtrain_scale = xtrain/255.0
xtest_scale = xtest/255.0

model = lr(solver = "saga", multi_class = "multinomial").fit(xtrain_scale, ytrain)

ypre = model.predict(xtest_scale)
accuracy = accuracy_score(ypre, ytest)
print(accuracy)

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        up_l = (int(width/2 - 50), int(height/2 - 50))
        low_r = (int(width/2 + 50), int(height/2 + 50))
        cv2.rectangle(gray, up_l, low_r, (0, 255, 0), 5)
        roi = gray[up_l[1]: low_r[1], up_l[0], low_r[0]]
        img_pil = Image.fromarray(roi)
        img_bw = img_pil.convert("L")
        img_bw_resize = img_bw.resize((28, 28), Image.ANTIALIAS)
        img_bw_resize_inverted = io.invert(img_bw_resize)
        pixelfilter = 20
        minpixel = np.percentile(img_bw_resize_inverted, pixelfilter)
        img_bw_resize_inverted_scaled = np.clip(img_bw_resize_inverted - minpixel, 0, 255)
        maxpixel = np.max(img_bw_resize_inverted)
        img_bw_resize_inverted_scaled = np.asarray(img_bw_resize_inverted_scaled)/maxpixel
        ts = np.array(img_bw_resize_inverted_scaled).reshape((1, 784))
        #ts = test sample
        #tp = test predict
        tp = model.predict(ts)
        print("Predicted class is: ", tp)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()