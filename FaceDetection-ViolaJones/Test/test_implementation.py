import cv2
import imutils
import math
import time
import json
import numpy as np
import FaceDetection

def read_model():
    with open('sample_model.json') as f:
        model = json.load(f)
        print(model)
    return model


def detect_face(img):
    integral_img = FaceDetector.get_integral_images(img)
    features = FaceDetector.calculate_and_merge_features(integral_img)
    model = read_model()
    for cascade in model:
        local_weak_classifiers = []
        feature_list = model[cascade]
        cascade_aplha = 0
        for feature in feature_list:
            index = feature['index']
            alpha = feature['alpha']
            cascade_aplha += alpha
            weak_clf = alpha * features[index]
            local_weak_classifiers.append(weak_clf)
        if(sum(local_weak_classifiers)>=(cascade_aplha * 0.5)):
            continue
        else:
            return False
    return True


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


windows = []
(winW, winH) = (24, 24)
for i in range(10):
    windowSize=(winW, winH)
    windows.append(windowSize)
    winW = math.floor(winW * 1.5)
    winH = math.floor(winH * 1.5)


image = cv2.imread("723.jpg",0)


for w in windows:
    print(w)
    (winW, winH) = w
    
    for (x, y, window) in sliding_window(image, stepSize=20, windowSize = (winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = image.copy()
        max_size = (24,24)
        resized = cv2.resize(window, (24, 24))
        
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
