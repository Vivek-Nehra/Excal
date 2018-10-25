import cv2
import pandas as pd
import numpy as np

def create(cells):
    L = []
    print("Enter Label : ")
    label = cv2.waitKey(0)
    if label == ord('n'):
        L.append(-1)
    L.append(label-48)
    cells = cv2.resize(cells,(28,28),interpolation = cv2.INTER_AREA)
    cells = cv2.cvtColor(cells,cv2.COLOR_BGR2GRAY)
    cells = cv2.Canny(cells,100,200)
    # cells = cv2.dilate(cells,np.ones([3,3]),iterations = 1)
    # cv2.imshow("Small",cells)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("Added : ", label)
    L.extend(cells.reshape(784).tolist())
    # print(L)
    return (L)


def write(dataset):
    # print(len(dataset[0]),dataset)
    df = pd.DataFrame(dataset)
    df.to_csv("Dataset.csv",sep = ',' , encoding = 'utf-8')
