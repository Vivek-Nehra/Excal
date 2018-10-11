import cv2
import pytesseract
import vertical

def row_height(lines):
    avg_distance = 0
    count=0
    if lines is None:
        return 0
    else:
        for idx,line in enumerate(lines):
            if idx == len(lines)-1:
                break

            avg_distance +=  max(lines[idx+1][0][1],lines[idx+1][0][3]) - min(line[0][1],line[0][3])
            count += 1

        avg_distance /= count
        return int(avg_distance)


def create_row(img,lines):
    row_size = row_height(lines)
    print("Row Height : " ,row_size)
    row,col = img.shape[:-1]

    for index,line in enumerate(lines) :
        if index == len(lines)-1:
            # roi = img[line[0][1]:row,:,:]
            break
        else:
            roi = img[min(line[0][1],line[0][3]):max(lines[index+1][0][1],lines[index+1][0][3]),:,:]

        roi_row,roi_col,_ = roi.shape
        #print(roi_row)
        if roi_row > 10:
        # if roi_row > row_size :
        	# print(pytesseract.image_to_string(in_im))
        	# vertical.detect(roi)
        	cv2.imshow("ROI",roi)
        	cv2.waitKey(0)
