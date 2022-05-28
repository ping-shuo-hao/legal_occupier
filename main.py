import numpy
import math
import cv2
import worker_classifier
from sklearn.linear_model import LogisticRegression

#Ramsey:[(10,396),(86,574),(581,283),(453,268)]
#Ashland:[(424,702),(555,872),(1588,484),(1513,468)]
#Thomasville South: [(34,855),(1289,643),(1534,644),(398,1076)]
#Thomasvile North: [(301,620),(1391,1055),(1734,761),(535,587)]

def getTrajectory(line):
    temp=""
    for char in line:
        if char!=" ":
            temp=temp+char
    arr=[]
    num=0
    po=1
    xValue=""
    yValue=""
    for char in temp:
        if char=='[':
            num+=1
        if char==']':
            num-=1
            po=1
            if num==1:
                x=int(xValue)
                y=int(yValue)
                arr.append([x,y])
            xValue=""
            yValue=""
        if num==2:
            if char == ',':
                po = 2
            if char.isdigit():
                if po==1:
                    xValue=xValue+char
                else:
                    yValue=yValue+char
    return arr

def changeTrajectoryFormat(trajectory):
    re=[]
    temp=[0,0]
    position=1
    for trj in trajectory:
        if position%2==0:
            temp[1]=trj
            re.append(temp)
        else:
            temp[0]=trj
        position+=1
    return re

def between(a,b,value):
    if a>b:
        if a>=value and value>=b:
            return True
        else:
            return False
    else:
        if a<=value and value<=b:
            return True
        else:
            return False

def checkMotion(trajectory):
#    trajectory=changeTrajectoryFormat(trajectory)
    dict={0:2,1:7,2:6,3:5,4:4,5:3,6:2,7:1,8:0}
    motion=[0,0,0,0,0,0,0,0,0]
    limit=len(trajectory)
    for i in range(1,limit):
        pre=trajectory[i-1]
        cur=trajectory[i]
        if cur[0]>pre[0]:
            if cur[1]>pre[1]:
                motion[8]+=1
            elif cur[1]==pre[1]:
                motion[5]+=1
            else:
                motion[2]+=1
        elif cur[0]==pre[0]:
            if cur[1]>pre[1]:
                motion[7]+=1
            elif cur[1]==pre[1]:
                motion[4]+=1
            else:
                motion[1]+=1
        else:
            if cur[1]>pre[1]:
                motion[6]+=1
            elif cur[1]==pre[1]:
                motion[3]+=1
            else:
                motion[0]+=1
    max=0
    seoncdMax=-1
    seoncdMaxValue=0
    start=trajectory[0]
    end=trajectory[-1]
    predictMotion=-1
    if end[0]>start[0]:
        if end[1]>start[1]:
            predictMotion=8
        elif end[1]==start[1]:
            predictMotion=5
        else:
            predictMotion=2
    elif end[0]==start[0]:
        if end[1]>start[1]:
            predictMotion=7
        elif end[1]==start[1]:
            predictMotion=4
        else:
            predictMotion=1
    else:
        if end[1]>start[1]:
            predictMotion=6
        elif end[1]==start[1]:
            predictMotion=3
        else:
            predictMotion=0
    for i in range(9):
        if i!=4:
            if motion[i]>motion[max]:
                max=i
    for i in range(9):
        if i != 4:
            if motion[i] > seoncdMaxValue and i!=max:
                seoncdMax = i
                seoncdMaxValue=motion[i]
    print(motion)
    print(len(trajectory))
    if motion[4]/len(trajectory)>0.5:
        return True
    if predictMotion!=-1:
        return motion[seoncdMax]!=motion[dict[max]]
#    dif=motion[max]-motion[seoncdMax]
#    threshold=dif/(len(trajectory)-1)
#    if predictMotion!=-1:
#        return (max==predictMotion and threshold>0.1)
    return False

def training(X,y):
    lm=LogisticRegression()
    result=lm.fit(X,y)
    return result.intercept_,result.coef_

def checkLegalOccupierForCarTruck(ROI,trajectory,le):
    if not checkMotion(trajectory):
        return False
    a=ROI[0]
    b=ROI[1]
    c=ROI[2]
    d=ROI[3]
    p1=[(a[0]+b[0])/2,(a[1]+b[1])/2]
    p2=[(b[0]+c[0])/2,(b[1]+c[1])/2]
    p3=[(c[0]+d[0])/2,(c[1]+d[1])/2]
    p4=[(a[0]+d[0])/2,(a[1]+d[1])/2]
    dis1=math.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)
    dis2=math.sqrt((p2[0]-p4[0])**2+(p2[1]-p4[1])**2)
    dis=0
    trackDirection=[]
    bound1=[a[0]-c[0],a[1]-c[1]]
    bound2=[b[0]-d[0],b[1]-d[1]]
    if dis1>dis2:
        dis=dis1
        trackDirection=[p1[0]-p3[0],p1[1]-p3[1]]
#        bound1=[a[0]-d[0],a[1]-d[1]]
#        bound2=[b[0]-c[0],b[1]-c[1]]
    else:
        dis=dis2
        trackDirection=[p2[0]-p4[0],p2[1]-p4[1]]
#        bound1=[a[0]-b[0],a[1]-b[1]]
#        bound2=[c[0]-d[0],c[1]-d[1]]
    dist1=math.sqrt(bound1[0]**2+bound1[1]**2)
    dist2=math.sqrt(bound2[0]**2+bound2[1]**2)
#    if numpy.dot(trackDirection,bound1)<0:
#        bound1=[-bound1[0],-bound1[1]]
#    if numpy.dot(trackDirection,bound2)<0:
#        bound2=[-bound2[0],-bound2[1]]
    limit=min(abs(numpy.dot(trackDirection,bound1)/(dis*dist1)),abs(numpy.dot(trackDirection,bound2)/(dis*dist2)))
    limit=limit-0.001
    #start={trajectory[0],trajectory[1]}
    #start={trajectory[-2],trajectory[-1]}
#    trajectory=changeTrajectoryFormat(trajectory)
    start=trajectory[0]
    end=trajectory[-1]
    if start[0]==end[0] and start[1]==end[1]:
        return False
    if len(trajectory)<le:
        return False
    objectVector = [end[0]-start[0],end[1]-start[1]]
    objectdis=math.sqrt(objectVector[0]**2+objectVector[1]**2)
#    print(f'object distance: {objectdis}')
    if objectdis<math.sqrt(10):
        return False
    angle=abs(numpy.dot(objectVector,trackDirection)/(objectdis*dis))
    print("limit: "+ str(limit))
    print("angle: " + str(angle))
    return between(limit,1,angle)

def find_bbox(link):
    video = cv2.VideoCapture(link)
    ret, frame = video.read()
    if ret==False:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#    cv2.imwrite('frame.jpg',frame)
    lower_bound, upper_bound = numpy.array([0, 200, 200]), numpy.array([3, 255, 255])
    mask1 = cv2.inRange(hsv, lower_bound, upper_bound)
    blur=cv2.GaussianBlur(mask1,(5,5),0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    #mask = cv2.bitwise_and(mask1, mask1)
#    cv2.imwrite("mask1.jpg", mask1)
    my_img=cv2.boundingRect(thresh)
    x, y, w, h = my_img
    if w*h < 100 or h<28:
        return False
#    ROI=frame[y:y + h, x:x+w]
    ROI = frame[y+23:y + h-5, x:x + w]
#    cv2.imwrite('t1.jpg',ROI)
    hsv=cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv, lower_bound, upper_bound)
    gray = cv2.bitwise_not(mask)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 15, -2)
    vertical = numpy.copy(bw)
    rows = vertical.shape[0]
    verticalsize = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical = cv2.bitwise_not(vertical)
#    cv2.imwrite('show.jpg',vertical)
#    print(vertical[0])
    minimum_len=int(0.4*len(vertical))
    sim=numpy.zeros((len(vertical),len(vertical[0])))
    for i in range(len(vertical)):
        for j in range(len(vertical[i])):
            if vertical[i][j]==0:
                sim[i][j]=1
    sim=sim.sum(axis=0)
    for i in range(len(vertical)):
        for j in range(len(vertical[i])):
            if sim[j]<minimum_len:
                vertical[i][j]=255
#    cv2.imwrite('show2.jpg', vertical)
    lst=[]
    for i in range(len(vertical)):
        meet_zero=False
        trigger = False
        sub=[]
        start = 0
        end = 0
        for j in range(len(vertical[i])):
            if vertical[i][j]==0:
                meet_zero=True
            if meet_zero and (not trigger and vertical[i][j]>0):
                trigger=True
                start=j
            if meet_zero and (trigger and vertical[i][j]==0):
                trigger=False
                end=j
                th=min(10,int(0.3*len(vertical[i])))
                if end-start>th:
                    sub.append((start,end))
        if len(sub)>0:
            maxlen=sub[0]
            for element in sub:
                if (element[1]-element[0])>(maxlen[1]-maxlen[0]):
                    maxlen=element
            lst.append(maxlen)
    if len(lst)==0:
        return False
    tar_center=lst[0]
    for element in lst:
        if (element[1]-element[0])<(tar_center[1]-tar_center[0]):
            tar_center=element
#    print(vertical)
    location=ROI[:,tar_center[0]:tar_center[1]]
#    cv2.imwrite('tar.jpg',location)
    video.release()
    return location

def checkLegalOccupierForPerson(img,thres1=0.15,thres2=0.13):

    return worker_classifier.classifer(img,thres1,thres2)

def findLegalOccupier(type,ROI,trajectory,link):
    type=type.lower()
    if type=="person":
        return checkLegalOccupierForPerson(find_bbox(link))
    else:
        return checkLegalOccupierForCarTruck(ROI,trajectory,5)


#c=0
#with open('tn.txt','r') as input_file:
#    lines=input_file.readlines()
#    for line in lines:
#        if findLegalOccupier('car',[(301,620),(1391,1055),(1734,761),(535,587)],getTrajectory(line),None):
#            c+=1
#    print(c)
        #print()
print(checkLegalOccupierForPerson(find_bbox('https://igct.s3.amazonaws.com/5fee675583dbb051a5aa4b16_1609536281.mp4')))