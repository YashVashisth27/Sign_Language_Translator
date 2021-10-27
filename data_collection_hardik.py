import cv2
import os
import numpy as np 

'''################################# GLOBAL VARIABLES #################################'''

#all directories
og_path = 'self_made_data/image_train/'
test_path = 'self_made_data/image_test/'
processed_path = 'self_made_data/image_train_processed/'
processed_test_path = 'self_made_data/image_test_processed/'

# directory list 
dir_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','0']


# file count in each directory 

file_count = {
    '1' : 0,
    '2' : 0,
    '3' : 0,
    '4' : 0,
    '5' : 0,
    '6' : 0,
    '7' : 0,
    '8' : 0,
    '9' : 0,
    'A' : 0,
    'B' : 0,
    'C' : 0,
    'D' : 0,
    'E' : 0,
    'F' : 0,
    'G' : 0,
    'H' : 0,
    'I' : 0,
    'J' : 0,
    'K' : 0,
    'L' : 0,
    'M' : 0,
    'N' : 0,
    'O' : 0,
    'P' : 0,
    'Q' : 0,
    'R' : 0,
    'S' : 0,
    'T' : 0,
    'U' : 0,
    'V' : 0,
    'W' : 0,
    'X' : 0,
    'Y' : 0,
    'Z' : 0,
    '0':0
}

'''################################# FUNCTIONS #################################'''

#creating all folders in directory if doesnot exist       
def create_all_required_directories(path):

    print(path)
    for i in dir_list:
        if i in os.listdir(path):
            print('yes')
        else:
            mode = 0o666
            os.mkdir(path+i, mode)
            
            
            
#  processing image 
def processing(img):

    minValue = 70
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return res

def mask_processing(roi):

    hsvim = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    
    # mask = processing(roi)
    masked = cv2.bitwise_and(roi, roi, mask=thresh)
    # cv2.imshow('mask',mask)
    # cv2.imshow('masked',masked)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)

    return thresh,masked,gray

# counts files in all directories and store in file_count dictionary
def count_files_dir(arg):
    # path = og_path+str(arg)+'\\'
    path = og_path+str(arg)
    for files in os.listdir(path):
        file_count[arg] = file_count[arg] + 1
        

def give_contours(roi):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(roi, [contours], -1, (255,255,0), 2)
    cv2.imshow("contours", roi)

    hull = cv2.convexHull(contours)
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
    cv2.imshow("hull", roi)  

# takes input such as entered key and roi of image and write the image in specific directory      
def show_info(k,roi):
    detail = " "
    # print("In function")
    for i in dir_list:
        # print(i , k , k%256, ord(i))
        if str(k%256) == str(ord(i)):
            detail = '{}_{}.png'.format(i,file_count[i]+201)
            spec_path = og_path+str(i)+'/'+ str(detail)
            spec_path = str(spec_path)
            
            print(spec_path)
            cv2.imwrite(spec_path, roi)
            file_count[i] = file_count[i]+1
            # print(detail)
            break
       
            
    return i,detail


'''################################# MAIN #################################'''

        
# calling count_files_dir()
# for i in dir_list:
#     count_files_dir(i)

# print(file_count)
    
    
# calling function for creating directories
create_all_required_directories(og_path)
create_all_required_directories(test_path)
create_all_required_directories(processed_path)
create_all_required_directories(processed_test_path)


cam=cv2.VideoCapture(0)
button_pressed='+'
upper_left=(0,100)
bottom_right=(250,350)




            


while True:
    ret,frame=cam.read()
    cv2.rectangle(img=frame,pt1=upper_left,pt2=bottom_right,color=(255,0,0),thickness=1)
    roi=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        
   
    if not ret:
        break
    
    k=cv2.waitKey(1)
    if k%256 ==27:
        print('Exiting the setup....')
        break
    else:
        # print(k%256)
        # print("key pressed")
        button_pressed,show_text = show_info(k-32,roi)
    

    
    fonts=cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(img=frame,text=show_text,org=(10,370),fontFace=fonts,fontScale=1,color=(255,0,0),thickness=1)
    
    cv2.imshow('Webcam',frame)
    cv2.imshow('roi',roi)
        
    mask = processing(roi)
    thresh,masked,gray = mask_processing(roi)
        
    cv2.imshow('mask',mask)
    cv2.imshow('masked',masked)
    cv2.imshow('thresh',thresh)
    cv2.imshow('gray',gray)
    # cv2.imshow('mask2',mask2)
    try:
        give_contours(roi)
    except:
        pass



cam.release()
cv2.destroyAllWindows()