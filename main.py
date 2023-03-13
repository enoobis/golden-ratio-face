import cv2

# Load haarcascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

# Load image
img = cv2.imread('images/4.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop through detected faces
for (x, y, w, h) in faces:
    # Draw rectangle around face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
    for (ex, ey, ew, eh) in eyes:
        # Draw rectangle around eyes
        cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
    # Detect nose
    nose = nose_cascade.detectMultiScale(gray[y:y+h, x:x+w])
    for (nx, ny, nw, nh) in nose:
        # Draw rectangle around nose
        cv2.rectangle(img, (x+nx, y+ny), (x+nx+nw, y+ny+nh), (0, 0, 255), 2)
        
    # Detect mouth
    mouth = mouth_cascade.detectMultiScale(gray[y:y+h, x:x+w])
    for (mx, my, mw, mh) in mouth:
        # Draw rectangle around mouth
        cv2.rectangle(img, (x+mx, y+my), (x+mx+mw, y+my+mh), (255, 255, 0), 2)

    # Calculate distances between features
    eye_dist = eyes[1][0] - eyes[0][0]
    nose_dist = nose[0][3]
    mouth_dist = mouth[0][1] - (eyes[0][1] + eyes[0][3])
    
    # Calculate ratios between features
    eye_to_nose_ratio = eye_dist / nose_dist
    nose_to_mouth_ratio = nose_dist / mouth_dist
    
    # Calculate overall ratio
    golden_ratio = 1.618
    face_ratio = eye_to_nose_ratio / golden_ratio * nose_to_mouth_ratio
    pst = face_ratio * 100
    
    # Print ratio for this face
    print('Golden face ratio: ', pst ,"%" )

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()