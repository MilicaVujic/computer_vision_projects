import numpy as np
import cv2 # OpenCV
import matplotlib
from matplotlib import pyplot as plt
# iscrtavanje slika u notebook-u

# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16,12

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

answers=[]
for i in range (1,11):

    img = load_image("pictures1/picture_"+str(i)+".jpg")
    i=img

   

    img=image_gray(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.erode(img, kernel, iterations=4)
    #display_image(img)

    img=image_bin(img)

    img = cv2.dilate(img, kernel, iterations=4)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    img=cv2.dilate(img,kernel,iterations=3)
    img=cv2.erode(img,kernel,iterations=1)


    img=cv2.dilate(img,kernel,iterations=2)
    img=cv2.erode(img,kernel,iterations=6)
    


    #display_image(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    #cv2.drawContours(i, contours, -1, (255, 0, 0), 1) 


    contours_ = [] 

    for contour in contours: 
        center, size, angle = cv2.minAreaRect(contour) 
        height, width = size

       
        if width > 21 and width < 70 and height>13.435 and height < 60:
            contours_.append(contour) 


    cv2.drawContours(i, contours_, -1, (255, 0, 0), 1)
    #print(len(contours_))
    answers.append(len(contours_))

    plt.imshow(i)
    plt.show()



correct_answers=[4,8,6,8,8,4,6,6,6,13]
sum=0
for i in range (0,10):
    print("picture_"+str(i+1)+".jpg-"+str(correct_answers[i])+"-"+str(answers[i]))
    sum+=np.abs(correct_answers[i]-answers[i])
print(sum/10)