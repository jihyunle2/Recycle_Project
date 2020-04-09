
from django.shortcuts import render,redirect
from time import sleep
import os
import cv2
import sys

count_other=0
count_hdpe=0
count_pet=0
count_can=0

def home(request):
    return render(request, 'home.html')

def camera_img(request):
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    os.chdir('/Users/Jihyun/Downloads/Recycle_test/recycleproject/recycleapp/static/img')

    while True:
        
        check, frame = webcam.read()
        key = cv2.waitKey(1)
        sleep(3)

        if key == ord('q'): #quit
            print("Turning off camera.")
            webcam.release()
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        else:
            cv2.imwrite('saved_img.jpg', img=frame)
            webcam.release()
            print("Image saved")
            break

    import retrain_Run_inference as r
    global count_can
    global count_pet
    a=[]
    b=[]
    a,b=r.run_inference_on_image()
    maxValue = b[0]
    index=0
    for i in range(1, len(a)):
        if maxValue < b[i]:
            maxValue = b[i]
            index=i

    maxValue*=100
    maxValue=round(maxValue,2)
    if a[index]=="b'can\\n'":
        count_can+=1
        return render(request,'result_can.html',{'percent':maxValue})
    elif a[index]=="b'pet\\n'":
        count_pet+=1
        return render(request,'result_pet.html',{'percent':maxValue})
    return render(request, 'test.html',{'a':a,'b':b})

def camera_mark(request):
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    os.chdir('/Users/Jihyun/Downloads/Recycle_test/recycleproject/recycleapp/static/img')

    while True:
        
        check, frame = webcam.read()
        key = cv2.waitKey(1)
        sleep(3)

        if key == ord('q'): #quit
            print("Turning off camera.")
            webcam.release()
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        else:
            cv2.imwrite('saved_img.jpg', img=frame)
            webcam.release()
            print("Image saved")
            break

    import letter as l

    L=l.letter_detect()

    global count_hdpe
    global count_other

    if L!=None and "OTHER" in L:
        count_other+=1
        return render(request,'result_other.html')
    elif L!=None and "HDPE" in L:
        count_hdpe+=1
        return render(request,'result_hdpe.html') 
    else:
        return render(request,'result_error.html')

    return render(request, 'test.html')

def letter(request):
    import letter as l
    L=l.letter_detect()
    return render(request,'about.html',{'L':L})

def about(request):
    return render(request,'about.html')

def result_count(request):
    global count_can
    global count_pet
    global count_hdpe
    global count_other
    return render(request,'result_count.html',{'pet':count_pet,'can':count_can,'other':count_other,'hdpe':count_hdpe})

def result_can(request):
    return render(request,'result_can.html')

def result_pet(request):
    return render(request,'result_pet.html')

def result_hdpe(request):
    return render(request, 'result_hdpe.html')

def result_other(request):
    return render(request, 'result_other.html')

def result_error(request):
    return render(request, 'result_error.html')

