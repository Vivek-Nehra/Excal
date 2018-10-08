import cv2


def row_height(lines):
    avg_distance = 0
    count=0
    for idx,line in enumerate(lines):
        if idx == len(lines)-1:
            break

        avg_distance +=  lines[idx+1][0][1] - line[0][1]
        count += 1

    avg_distance /= count
    return int(avg_distance)


def create_row(img,lines):
    row_size = row_height(lines)
    print(row_size)
    row,col = img.shape[:-1]

    for index,line in enumerate(lines) :
        if index == len(lines)-1:
            # roi = img[line[0][1]:row,:,:]
            break
        else:
            roi = img[max(line[0][1],line[0][3]):max(lines[index+1][0][1],lines[index+1][0][3]),:,:]

        roi_row,roi_col,_ = roi.shape
        #print(roi_row)
        # if roi_row > 10:
        if roi_row > row_size :
            cv2.imshow("ROI",roi)
            cv2.waitKey(0)
