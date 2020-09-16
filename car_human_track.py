import cv2

img_file='testcar.jpg'
#video=cv2.VideoCapture('testvideo.mp4')
#video=cv2.VideoCapture('testvideo1.mp4')
video=cv2.VideoCapture('testvideo2p.mp4')
car_tracker_file='car_detector.xml'
pedestrian_tracker_file='haarcascade_fullbody.xml'

car_tracker= cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker= cv2.CascadeClassifier(pedestrian_tracker_file)
while True:
    (read_successful, frame)=video.read()
    if read_successful:
        gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    cars=car_tracker.detectMultiScale(gray_frame)
    pedestrians=pedestrian_tracker.detectMultiScale(gray_frame)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x+1,y+2),(x+w, y+h), (255,0,0), 2)
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255), 2)
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,255), 2)
    cv2.imshow('CAR and PEDESTRIAN DETECTOR',frame)
    key= cv2.waitKey(1)
    if key==81 or key==113:
        break
video.release()
"""
img= cv2.imread(img_file)
car_tracker= cv2.CascadeClassifier(classifier_file)
black_n_white=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars=car_tracker.detectMultiScale(black_n_white)
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,0,255), 2)

cv2.imshow('CAR DETECTOR',gray_frame)
cv2.waitKey(1)
"""
print('code done')
