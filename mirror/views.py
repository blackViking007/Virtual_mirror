from django.http import HttpResponse
from django.shortcuts import render_to_response
from django.http import HttpResponseRedirect
from django.shortcuts import render
import cv2

def welcome(request):
    return render_to_response('welcome.html')

def caminit(request):
    cap = cv2.VideoCapture(0)


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img1 = frame
        #print "img1 shape", img1.shape
        img2 = cv2.imread('fb.png',1)
        rows,cols,channels = img2.shape
        glassesWidth = 300
        glassesHeight = 400
        glasses = cv2.resize(img2, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
        # I want to put logo on top-left corner, So I create a ROI
        rows_,cols_,channels_ = glasses.shape
        #print "mask shape", glasses.shape
        roi = img1[40:40+rows_, 160:160+cols_]
        #print "roi size", roi.shape
        #cv2.rectangle(roi,(EX[0],sumy),(EX[0]+2,sumy+2),(255,0,0),1)

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(glasses,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('1.img2gray',img2gray)
        #cv2.waitKey(0)
        ret, mask = cv2.threshold(img2gray, 148, 255, cv2.THRESH_BINARY)
        #cv2.imshow('1.mask',mask)
        #cv2.waitKey(0)
        mask_inv = cv2.bitwise_not(mask)
        #cv2.imshow('1.mask_inv',mask_inv)
        #cv2.waitKey(0)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
        #cv2.imshow('2.messipart_roi_blackout_seat',img1_bg)
        #cv2.waitKey(0)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(glasses,glasses,mask = mask_inv)
        #cv2.imshow('3.and_op_for_clourfullogo',img2_fg)
        #cv2.waitKey(0)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        img1[40:40+rows_, 160:160+cols_] = dst
        # Display the resulting frame
        cv2.imshow('frame',img1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
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
    print "faceCascade data type",type(faceCascade)

    # Read the image
    image = cv2.imread("snap.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray"+".jpg",gray)
    graymax = cv2.imread("gray.jpg")

    roi_graymax = graymax[0:0+30, 0:0+30]
    print "/////////////////////roi_graymax//////////////////////",roi_graymax
    #cv2.rectangle(graymax, (0, 0), (0+30, 0+30), (0, 255, 0), 2)
    cv2.imshow('res',graymax)
    cv2.waitKey(0) 




    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        graymax,
        scaleFactor=1.0000000001,
        minNeighbors=3,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print "Found {0} faces!".format(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        print "xxxxxxxxxxxxxxxxxxxx",x
        print "yyyyyyyyyyyyyyyyyyyy",y
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        print "roi_color",roi_color
        print "shape of roi",roi_color.shape
        eyes = eye_cascade.detectMultiScale(roi_gray)
        sumh = 0  
        sumy = 0
        sumw = 0
        print "eyes  before removal of third eye",eyes
        print len(eyes)
        EX = []
        EW = []
        realeyes = []
        EX_ = []
        if len(eyes) > 2:
            for (ex,ey,ew,eh) in eyes:
              EX_.append(ex)
              realeyes.append((ex,ey,ew,eh))
            print "width array including 3rd eye",EX_
            print "realeyes array still with 3rd eye",realeyes

            diff1 = abs(EX_[1]-EX_[0])
            diff2 = abs(EX_[2]-EX_[0])
            diff3 = abs(EX_[1]-EX_[2])
            maxdiff = max(diff3,diff2,diff1)
            if maxdiff == diff1:
                realeyes.pop(2)
            elif maxdiff == diff2:
                realeyes.pop(1)
            elif maxdiff == diff3:
                realeyes.pop(0)


        else:
          realeyes = eyes
        
        print "realeyes after removal of 3rd eye",realeyes

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

    img2 = cv2.imread('g2.jpg')
    rows,cols,channels = img2.shape
    glassesWidth = 10*glassew/9
    glassesHeight = glassesWidth * rows / cols
    x1 = glassex - (glassesWidth/4)
    x2 = glassex + glassew + (glassesWidth/4)
    y1 = glassey + glasseh - (glassesHeight/2)
    y2 = glassey + glasseh + (glassesHeight/2)

    glassesWidth = x2 - x1
    glassesHeight = y2 - y1

    img1 = image 

    glasses = cv2.resize(img2, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
    xtemp = x - 5*glassesHeight/glassesWidth
    ytemp = y + 3*glassew/8
    # I want to put logo on top-left corner, So I create a ROI
    rows_,cols_,channels_ = glasses.shape

    roi = img1[xtemp:xtemp+rows_, ytemp:ytemp+cols_]
    #cv2.rectangle(roi,(EX[0],sumy),(EX[0]+2,sumy+2),(255,0,0),1)

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(glasses,cv2.COLOR_BGR2GRAY)
    cv2.imshow('1.img2gray',img2gray)
    cv2.waitKey(0)
    ret, mask = cv2.threshold(img2gray, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow('1.mask',mask)
    cv2.waitKey(0)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('1.mask_inv',mask_inv)
    cv2.waitKey(0)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    cv2.imshow('2.messipart_roi_blackout_seat',img1_bg)
    cv2.waitKey(0)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(glasses,glasses,mask = mask_inv)
    cv2.imshow('3.and_op_for_clourfullogo',img2_fg)
    cv2.waitKey(0)
    cv2.imwrite("fore.jpg",img2_fg)
    img2_fg_we = img2_fg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    erosion = cv2.erode(img2_fg_we,kernel,iterations = 1)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,erosion)
    img1[xtemp:xtemp+rows_, ytemp:ytemp+cols_] = dst

    cv2.imshow('res',img1)
    cv2.imwrite("final.jpg",img1)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()