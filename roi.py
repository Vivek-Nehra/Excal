import sys
import cv2
import pytesseract
import vertical
from keras.models import load_model
import rectangle
import Cell_test
#import Predict
import pandas as pd

def row_height(lines):          # Find average height of each row
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


def create_row(img,horizontal_lines,vertical_lines):        # Create Cells
    row_size = row_height(horizontal_lines)
    print("Row Height : " ,row_size)
    row,col = img.shape[:-1]
    counter = 0
    model = load_model('Classifier/PA.h5')         # Load Model
    output=[]
    header = [["Roll No","Name "]]
    max_col_count = 0
    TA = []
    for index,line in enumerate(horizontal_lines) :
        col_count = 0
        if index == len(horizontal_lines)-1:
            # roi = img[line[0][1]:row,:,:]
            break
        else:
            roi = img[min(line[0][1],line[0][3]):max(horizontal_lines[index+1][0][1],horizontal_lines[index+1][0][3]),:,:]      # Create rows

        roi_row,roi_col,_ = roi.shape
        if roi_row > 10:
        # if roi_row > row_size :
            # cv2.imshow("ROI",roi)
            # cv2.waitKey(0)

            counter += 1
            if counter > 3:
                # print()
                row = []
                cnt = 0


                current_attendance = 0
                for idx,ver_line in enumerate(vertical_lines):
                    if idx == len(vertical_lines)-1:
                        break
                    if cnt == 0:
                        cnt += 1
                        continue

                    cells = roi[:,ver_line:vertical_lines[idx+1]]           # Detect Cells in each row
                    cv2.imwrite("temp_img.jpg",cells)
                    # cv2.imshow("Cells",cells)
                    # cv2.waitKey(0)
                    text = Cell_test.printed_text("temp_img.jpg",model)     # Get text in the cells
                    # print (text,end=' ')
                    if text != "":
                        col_count += 1
                        row.append(text)

                       
                    if text == '1':
                        current_attendance += 1

                    # if cv2.waitKey(0) == ord('q'):
                        # sys.exit(0)

                if max_col_count < col_count:
                    max_col_count = col_count

                TA.append([current_attendance])

    # Create Output Files
                output.append(row)


    for i in range(1,max_col_count-1):
        header[0].extend([i])

    header[0].extend(["Total Attendance"])
    print("Total Columns : " ,max_col_count)

    for i in range(len(TA)):
        TA[i][0] = str(round(TA[i][0]/(max_col_count-2) * 100,2))
        TA[i][0] += "%"

    for i in range(len(output)):
        cur_len = len(output[i])
        if cur_len != max_col_count:
            for j in range(max_col_count-cur_len):
                output[i].append('A')


    for i in range(len(output)):
        output[i].append(TA[i][0])
    

    header.extend(output)




    # print(header)
    df = pd.DataFrame(header)
    # print(df)
    df.to_csv("Output/Output.csv",sep=' ',encoding='utf_8',header=False,index=False,na_rep = '')

    df_new = pd.read_csv("Output/Output.csv",sep=' ')
    writer = pd.ExcelWriter("Output/Result.xlsx")
    df_new.to_excel(writer,index=False)
    writer.save()
    print("Output Generated")