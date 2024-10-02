import numpy as np
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
import sys
import csv



def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

if len(sys.argv)<2:
    putanja='data1/'
else:
    putanja=sys.argv[1]
    

ipos = load_image(putanja+'pictures/p_1.png')
visina,sirina=ipos.shape


pos_imgs = []
neg_imgs = []

for i in range (1,308):
    img = load_image(putanja+"pictures/p_"+str(i)+".png")   
    pos_imgs.append(img)
for i in range (1,317):
    img = load_image(putanja+"pictures/n_"+str(i)+".png")   
    neg_imgs.append(img)
        

def get_hog():
    # Racunanje HOG deskriptora za slike iz MNIST skupa podataka
    img_size = (visina, sirina)
    nbins = 9
    cell_size = (8, 8)
    block_size = (2, 2)
    hog = cv2.HOGDescriptor(_winSize=(ipos.shape[1] // cell_size[1] * cell_size[1],
                                      ipos.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog


pos_features = []
neg_features = []
labels = []

for img in pos_imgs:
    pos_features.append(get_hog().compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(get_hog().compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


classifier = SVC(probability=True)
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)


def detect_line(img):
    img = cv2.GaussianBlur(img, (7, 7), 0) 

    edges_img = cv2.Canny(img, 190, 220, apertureSize=3)
    

    min_line_length=100
    
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=30)
    
    x1 = lines[0][0][0]
    y1 = 750 - lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = 750 - lines[0][0][3]


    
    return (x1, y1, x2, y2)

def classify_window(window):
    features = get_hog().compute(window).reshape(1, -1)
    return classifier.predict_proba(features)[0][1]

def non_max_suppression(boxes, scores, threshold):
    order = scores.argsort()[::-1]

    selected_boxes = []

    while order.size > 0:
        selected_box = boxes[order[0]]
        selected_boxes.append(selected_box)

        x1 = np.maximum(selected_box[0], boxes[order[1:]][:, 0])
        y1 = np.maximum(selected_box[1], boxes[order[1:]][:, 1])
        x2 = np.minimum(selected_box[2], boxes[order[1:]][:, 2])
        y2 = np.minimum(selected_box[3], boxes[order[1:]][:, 3])

        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_selected = (selected_box[2] - selected_box[0]) * (selected_box[3] - selected_box[1])
        area_others = (boxes[order[1:]][:, 2] - boxes[order[1:]][:, 0]) * (boxes[order[1:]][:, 3] - boxes[order[1:]][:, 1])
        iou = intersection / (area_selected + area_others - intersection)

        order = order[1:][iou <= threshold]

    return np.array(selected_boxes)
    


def process_video(video_path, hog_descriptor):
    # procesiranje jednog videa
    preslo = 0

    
    # ucitavanje videa
    frame_num = 1
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova
    
    # analiza videa frejm po frejm
    while True:
         # indeksiranje frejmova

        grabbed, frame = cap.read()

        # ako frejm nije zahvacen
        if not grabbed:
            break
        if frame is not None:
            frame=cv2.resize(frame,(1500,750))


        if frame_num == 1: # ako je prvi frejm, detektuj liniju
            line_coords = detect_line(frame)
            
            
            liney=line_coords[1]
        
        
        rectangles=[]
        rectangles_array=np.array([[1,2,3,4]])
        scores=np.array([1])
        for y in range(0, frame.shape[0], 10):
            for x in range(0, frame.shape[1], 10):
                if x>400 and x<1000 and y>175 and y<640:
                    this_window = (y, x,y+60,x+30)
                    new_vector = np.array([y,x,y+60,x+30])
                    window = frame[y:y+60, x:x+30]
                    window=cv2.resize(window,(60,120))
                    score = classify_window(window)
                    if score>0.99:
                        rectangles.append(this_window)
                        rectangles_array = np.concatenate([rectangles_array, [new_vector]])
                        scores=np.append(scores,score)


        rectangles_array=rectangles_array[1:]
        scores=scores[1:]

        rectangles=non_max_suppression(rectangles_array,scores,0.3)
        for i in range(len(rectangles)):
            y1=rectangles[i][0]
            x1=rectangles[i][1]
            y2=rectangles[i][0]+60
            x2=rectangles[i][1]+30

           
            center_x = x1 +  (x2-x1)/ 2
            center_y = 750 - (y1 + (y2-y1) / 2)
                
            if np.abs(liney-center_y)<27.23:
                preslo+=1



        frame_num += 7
        cap.set(1, frame_num)

    cap.release()
    return preslo
    

hog=get_hog()
data=[]
predictions=[]

with open(putanja+"/counts.csv",mode='r') as file:
    redovi=csv.reader(file)
    next(redovi)
    for red in redovi:
        data.append(int(red[1]))

for i in range(1,5):
    a=process_video(putanja+"videos/segment_"+str(i)+".mp4", hog)
    print("segment"+str(i)+".mp4-"+str(data[i-1])+"-"+str(a))
    predictions.append(a)

mae=mean_absolute_error(data,predictions)
print(mae)
