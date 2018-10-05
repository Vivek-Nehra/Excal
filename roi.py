import cv2

def create_roi(img,lines):
    for index,line in enumerate(lines) :
        if index == len(lines)-1:
            break
        roi = img[line[0][1]:lines[index+1][0][1],:,:]
        row,col,_ = roi.shape
        #print(row)
        if row > 15:
            cv2.imshow("ROI",roi)
            cv2.waitKey(0)
