import cv2
#import sys

# Get user supplied values

cam = cv2.VideoCapture()

def getframe(name):  
  cam.open(0)
  img=cam.read()
  cv2.imwrite(str(name)+".jpg",img[1])
  cam.release() 


getframe("snap")


# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Read the image
image = cv2.imread("snap.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Load the glass image
imgGlass=cv2.imread('glass.png',-1)

#Creating the mask for the glasses
#imgGlassgray=cv2.cvtColor(imgGlass,cv2.COLOR_BGR2GRAY)
#ret, glass_mask=cv2.threshold(imgGlassgray,0,255, cv2.THRESH_BINARY)
#Create inverted mask
glass_mask=imgGlass[:,:,3]
glass_mask_inv=cv2.bitwise_not(glass_mask)
#Convert glass image to BGR
#and save the original image size
imgGlass=imgGlass[:,:,0:3]
origGH, origGW=imgGlass.shape[:2]

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    sumh = 0  
    sumy = 0
    sumw = 0
    print "eeeeeeeeeyes",eyes
    print len(eyes)
    EX = []
    EW = []
    realeyes = []

    if len(eyes) > 2:
      for (ex,ey,ew,eh) in eyes:
        realeyes.append((ex,ey,ew,eh))
    else:
      realeyes = eyes

    for (ex,ey,ew,eh) in realeyes:
        EX.append(ex)
        EW.append(ew)
        sumh += eh
        sumw += ew
        print "sumhhhhhhhhhhhh",sumh
        sumy += ey
        print "summmmmmmmmmmmmyyyyyyyyy",sumy
        cv2.rectangle(roi_color,(ex,ey),(ex+2,ey+2),(0,255,0),3)
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
    #gen taking avg
    sumh = sumh/2
    sumy = sumy/2
    sumw = sumw/2
    print "sumhhhhhhhhhhhh",sumh
    print "summmmmmmmmmmmmyyyyyyyyy",sumy
    print "EXxxxxxxxxxxxxxxxx",EX
    print "EYYYYYYYYYYYYYYY",EW
    EX.sort()
    EW.sort()
    cv2.rectangle(roi_color,(EX[0],sumy),(EX[0]+(EX[1]-EX[0])+sumw,ey+sumh),(0,0,255),1)
    #if EW[1]>EW[0]:
      #if right eye is bigger 
    #  cv2.rectangle(roi_color,(EX[0],sumy),(EX[0]+(EX[1]-EX[0])+EW[1],ey+sumh),(0,0,255),1)
    #else:
      #cv2.rectangle(roi_color,(EX[0],sumy),(EX[0]+(EX[1]-EX[0])+sumw,ey+sumh),(0,0,255),1)


    # get avg of all para then used to draw rectangle
    glassex =  EX[0]
    glassey = sumy
    if EW[1]>EW[0]:
      glassew = (EX[1]-EX[0])+EW[1]
    else:
      glassew = (EX[1]-EX[0])+sumw
    glasseh = sumh
    
    print "glassex", glassex
    print "glassey", glassey
    print "glasseh", glasseh
    print "glassew", glassew
    #Creating Region Of Image (Red)
    roi_red = image[glassey:glassey+glasseh, glassex:glassex+glassew];

    #Masking the glasses onto the face
    #for (glassex ,glassey,glassew,glasseh) in eyes:
    print "In glass for loop"
    glassesWidth = 2*glassew
    glassesHeight = glassesWidth * origGH / origGW

    # Center the glasses on the bottom of the nose
    x1 = glassex - (glassesWidth/4)
    x2 = glassex + glassew + (glassesWidth/4)
    y1 = glassey + glasseh - (glassesHeight/2)
    y2 = glassey + glasseh + (glassesHeight/2)

    # Check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # Re-calculate the width and height of the glasses image
    glassesWidth = x2 - x1
    glassesHeight = y2 - y1

    # Re-size the original image and the masks to the glasses sizes
    # calcualted above
    glasses = cv2.resize(imgGlass, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(glass_mask, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(glass_mask_inv, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)

    # take ROI for glasses from background equal to size of glasses image
    #rows,cols,channels = imgGlass.shape
    roi = roi_color[y1:y2,x1:x2]
    print "ROI Red", roi_red
    print "ROI Color", roi_color
    print "ROI",roi

    # roi_bg contains the original image only where the glasses is not
    # in the region that is the size of the glasses.
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # roi_fg contains the image of the glasses only where the glasses is
    roi_fg = cv2.bitwise_and(glasses,glasses,mask = mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg,roi_fg)

    # place the joined image, saved to dst back over the original image
    roi_color[y1:y2, x1:x2] = dst





cv2.imshow("Faces found and eyes probably", image)
cv2.waitKey(0)
