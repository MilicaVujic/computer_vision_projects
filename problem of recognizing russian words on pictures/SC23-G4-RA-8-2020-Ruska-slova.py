import numpy as np
import cv2 # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
from scipy import ndimage
matplotlib.rcParams['figure.figsize'] = 16,12
# keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.cluster import KMeans

from tensorflow.keras.optimizers import SGD
import sys
import csv

if len(sys.argv)<2:
    putanja='data1/'
else:
    putanja=sys.argv[1]

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    

    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)

    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result

def display_result_without_spaces(outputs, alphabet):
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    for idx, output in enumerate(outputs[1:, :]):
        result += alphabet[winner(output)]
    return result

def select_roi_with_distances(image_orig, image_bin):
    a=image_bin.copy()
    contours, hierarchy = cv2.findContours(image_bin.copy(),  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sakvakomnacrtane=[]
    nacrtane=[]
    regions_array = []

    for contour in contours:
        for contour1 in contours :
            x, y, w, h = cv2.boundingRect(contour)  #gornja
            x1,y1,w1,h1=cv2.boundingRect(contour1)  #donja
            if (h>25 and w<60 ):
                if( y1+h1-y<5 and x<x1+w1<x+w):
                    if x in nacrtane:
                        visina=y+h-y1
                        region=image_bin[y1:y1+visina+1,x:x+w+1]
                        f=[region[1][0] for region in regions_array]
                        if x not in f:
                            regions_array.append([resize_region(region), (x, y1, w, visina)])
                            cv2.rectangle(image_orig, (x, y1), (x+w, y1+visina), (0, 255, 0), 2)
                            sakvakomnacrtane.append(x)
                    else:
                        visina=y+h-y1
                        region=image_bin[y1:y1+visina+1,x:x+w+1]
                        f=[region[1][0] for region in regions_array]
                        if x not in f:
                            regions_array.append([resize_region(region), (x, y1, w, visina)])
                            cv2.rectangle(image_orig, (x, y1), (x+w, y1+visina), (0, 255, 0), 2)
                            sakvakomnacrtane.append(x)
                else:
                    if x not in sakvakomnacrtane:
                        if proveri_slicne_x_koordinate(contours,x) == False:
                            region = image_bin[y:y+h+1, x:x+w+1]
                            f=[region[1][0] for region in regions_array]
                            if x not in f:
                                regions_array.append([resize_region(region), (x, y, w, h)])
                                cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                nacrtane.append(x)
  
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def proveri_slicne_x_koordinate(lista_kontura, ciljana_x_koordinata, tolerancija=5):
    for kontura in lista_kontura:
        x, y, w, h = cv2.boundingRect(kontura)
        if x!=ciljana_x_koordinata:
        # Provera sličnosti x-koordinata sa zadatom tolerancijom
            if abs(x - ciljana_x_koordinata) <= tolerancija:
                return True

    return False


def hammingDist(str1, str2):

    if len(str1)==len(str2):
        i = 0 
        count = 0
        while i!=len(str1):
        
            if (str1[i] != str2[i]):
                count=count+1
            i=i+1
    else:
        if len(str1)>len(str2):
            duzina=len(str2)
        else:
            duzina=len(str1)
        i=0
        count=0
        while i!=duzina:
            if(str1[i]!=str2[1]):
                count=count+1
            i=i+1
        count=count+np.abs(len(str1)-len(str2))
    
    return count


#kreiranje trening skupa
azbuka=[]

image_color = load_image(putanja+'pictures/captcha_1.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova=[]
slike_slova.append(letters[0])  #с
azbuka.append('с')
slike_slova.append(letters[1])  #т
azbuka.append('т')
slike_slova.append(letters[2])  #у
azbuka.append('у')
slike_slova.append(letters[3])  #ч
azbuka.append('ч')
slike_slova.append(letters[4])  #а
azbuka.append('а')
slike_slova.append(letters[6])  #ь
azbuka.append('ь')


image_color = load_image(putanja+'pictures/captcha_2.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[0]) #к
azbuka.append('к')
slike_slova.append(letters[2]) #о
azbuka.append('о')
slike_slova.append(letters[3]) #э
azbuka.append('э')

image_color = load_image(putanja+'pictures/captcha_3.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[2]) #ш
azbuka.append('ш')
slike_slova.append(letters[5]) #н
azbuka.append('н')
slike_slova.append(letters[6]) #и
azbuka.append('и')
slike_slova.append(letters[7]) #ц
azbuka.append('ц')

image_color = load_image(putanja+'pictures/captcha_4.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[0]) #ф
azbuka.append('ф')
slike_slova.append(letters[2]) #л
azbuka.append('л')
slike_slova.append(letters[3]) #д
azbuka.append('д')
slike_slova.append(letters[5]) #в
azbuka.append('в')
slike_slova.append(letters[7]) #й
azbuka.append('й')

image_color = load_image(putanja+'pictures/captcha_5.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[1]) #з
azbuka.append('з')

image_color = load_image(putanja+'pictures/captcha_6.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[0]) 
azbuka.append('б')
slike_slova.append(letters[1]) #е
azbuka.append('е')
slike_slova.append(letters[3]) #п
azbuka.append('п')

image_color = load_image(putanja+'pictures/captcha_7.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[0]) #ю
azbuka.append('ю')
slike_slova.append(letters[6]) #щ
azbuka.append('щ')

image_color = load_image(putanja+'pictures/captcha_8.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[0]) #я
azbuka.append('я')
slike_slova.append(letters[1]) #г
azbuka.append('г')
slike_slova.append(letters[5]) #ё
azbuka.append('ё')
slike_slova.append(letters[6]) #ж
azbuka.append('ж')

image_color = load_image(putanja+'pictures/captcha_9.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[9]) #р
azbuka.append('р')

image_color = load_image(putanja+'pictures/captcha_10.jpg')
image_color=image_color[170:270,200:850]
img = image_bin(image_gray(image_color))
img_bin = erode(dilate(img))
selected_regions, letters, region_distances = select_roi_with_distances(image_color.copy(), img_bin)
slike_slova.append(letters[0])#х
azbuka.append('х') 
slike_slova.append(letters[8]) #Ъ
azbuka.append('ъ')

#obucavanje mreze
inputs = prepare_for_ann(slike_slova)
outputs = convert_output(azbuka)
ann = create_ann(len(azbuka))
ann = train_ann(ann, inputs, outputs, epochs=2000)

#testiranje
# Definisanje praznog niza
reci = []
nazivi_slika=[]

# Otvori fajl i čitaj liniju po liniju
with open(putanja+'res.csv', 'r', encoding='utf-8') as fajl:
    next(fajl)
    for linija in fajl:
        # Razdvajanje linije na osnovu zareza
        delovi = linija.strip().split(',')
        reci.append(delovi[1])
        nazivi_slika.append(delovi[0])

suma_gresaka=0
for i in range (1,11):
    image_color = load_image(putanja+'pictures/captcha_'+str(i)+'.jpg')
    image_color=image_color[170:270,200:850]
    img = image_bin(image_gray(image_color))
    selected_regions, letters, distances = select_roi_with_distances(image_color.copy(), img)
    rastojanja=distances
    suma_distanci=0


    postoji_razmak=False
    prosek_distanci=np.average(distances)
    for distance in distances:
        if  np.abs(distance-prosek_distanci)>prosek_distanci:
            postoji_razmak=True

    if postoji_razmak==True:
        rastojanja = np.array(rastojanja).reshape(len(rastojanja), 1)
        k_means = KMeans(n_clusters=2,n_init=10)
        k_means.fit(rastojanja)
        inputs = prepare_for_ann(letters)
        results = ann.predict(np.array(inputs, np.float32))
        print(nazivi_slika[i-1]+"-"+reci[i-1]+"-"+display_result_with_spaces(results, azbuka, k_means))
        hd=hammingDist(display_result_with_spaces(results, azbuka, k_means),reci[i-1])
        suma_gresaka+=hd
    else:
        inputs = prepare_for_ann(letters)
        results = ann.predict(np.array(inputs, np.float32))
        print(nazivi_slika[i-1]+"-"+reci[i-1]+"-"+display_result_without_spaces(results, azbuka))
        hd=hammingDist(display_result_without_spaces(results, azbuka),reci[i-1])
        suma_gresaka+=hd


print(suma_gresaka)

