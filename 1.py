import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
import time


df=pd.read_csv("LidarData1.csv",sep=",")
df["3"]=0

df=np.array(df)
df_p=df.copy()

def shortest_distance(x1, y1, a, c):
    d = abs((a * x1 - y1 + c)) / (math.sqrt(a * a + 1))
    return d

def most_distance(x,y,a,c):
    max_d=shortest_distance(x[0], y[0], a, c)
    p_max=[x[0],y[0]]
    for i in range(len(x)):
        dist=shortest_distance(x[i], y[i], a, c)
        if dist>max_d:
            max_d=dist
            p_max=[x[i],y[i]]
    return [max_d,p_max]

def clusters1(max_dist_point,group_c,a,df,group):
    a1 = -1 / a
    b1 = max_dist_point[1] - max_dist_point[0] * a1
    ind_left, ind_right = 0, 0
    for i in range(len(df)):
        if df[i,3]  == group_c and df[i,2] < df[i,1] * a1 + b1:
            df[i,3] = group
            #ind_left = ind_left + 1
        elif df[i,3]  == group_c and df[i,2] >= df[i,1] * a1 + b1:
            df[i,3] = group+1
            #ind_right = ind_right + 1
    return df

def split(df,thresh1):
    cond=1
    group=1
    while cond:
        #print(group)
        clusters = set(df[:,3])
        for group_c in clusters:
            ind = np.where(df[:, 3] == group_c)
            df1 = df[ind]
            if len(df1) == 1:
                continue
            a, c = np.polyfit(df1[:, 1], df1[:, 2], 1)
            max = most_distance(df1[:, 1], df1[:, 2], a, c)
            if max[0] > thresh1:
                df = clusters1(max[1], group_c, a, df, group)

            else:
                cond = 0
            group += 2

    return df

def merge(df):
    clusters = set(df[:, 3])
    lines=[]
    for j in clusters:
        ind = np.where(df[:, 3] == j)
        df1 = df[ind]
        l=len(df1)
        a1, c1 = np.polyfit(df1[:, 1], df1[:, 2], 1)
        if l>10:
            lines.append([a1,c1,l,0])
    lines=np.array(lines)
    for j in range(len(lines)):
        for z in range(len(lines)):
            if j!=z:
                if abs(lines[j,0]-lines[z,0]) < 0.5 and abs(lines[j,1]-lines[z,1])<0.5:
                    if lines[j,2]>lines[z,2]:
                        lines[j, 3] = 1
                        lines[z,3]=0
                    else:
                        lines[j, 3] = 0
                        lines[z, 3] = 1
                else:
                    lines[j, 3] = 1
    ind = np.where(lines[:, 3] == 1)
    lines1 = lines[ind]
    return lines1

def regression(df,N):
    X = []
    for i in range(0, len(df)-N):
        df1 = df[i:i + N, :]
        a, b = np.polyfit(df1[:, 1], df1[:, 2], 1)
        X.append([a, b])
    X = np.array(X)
    db = DBSCAN(eps=0.5, min_samples=3).fit(X)
    groups = db.labels_
    j = 0
    for i in range(0, len(df) - N):
        df[i:i + N, 3] = groups[j]
        j = j + 1
    #visualization_results(dekart_coord, 'Results of Linear regression algorithm')
    return df

def f_lines(df):
    clusters = set(df[:, 3])
    lines=[]
    for j in clusters:
        if j!=-1:
            ind = np.where(df[:, 3] == j)
            df1 = df[ind]
            l = len(df1)
            a1, c1 = np.polyfit(df1[:, 1], df1[:, 2], 1)
            if l > 10:
                lines.append([a1, c1, l, 0])

    lines=np.array(lines)
    return lines

def hough_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.dilate(gray, kernel)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 45, minLineLength, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite('lines1.jpg', img)
    return

def plot(df,lines):
    for i in range(len(lines)):
        y = df[:, 1] * lines[i, 0] + lines[i, 1]
        x = df[:, 1]
        plt.plot(x, y)
    x = df[:, 1]
    y = df[:, 2]

img = cv2.imread('LidarData.png')
t0=time.time()
df1=split(df,0.09)
lines1=merge(df1)
ts=time.time()-t0
print('Absolute time of split and merge approach '+str(round(ts,2)))

df2=regression(df_p,7)
tr=time.time()-t0
print('Absolute time of Linear regression approach '+str(round(tr,2)))

hough_transform(img)
th=time.time()-t0
print('Absolute time of Hough Transform algorithm '+str(round(th,2)))


img2=cv2.imread('lines1.jpg')




lines2=f_lines(df2)

plt.subplot(321)
plot(df1,lines1)
plt.scatter(df[:, 1],df[:, 2],color="black")
plt.title("Split and merge approach")
plt.axis(xmin=-6, xmax=6, ymin=-6, ymax=6)

plt.subplot(322)
plot(df2,lines2)
plt.scatter(df[:, 1],df[:, 2],color="black")
plt.title("Linear regression")
plt.axis(xmin=-6, xmax=6, ymin=-6, ymax=6)

plt.subplot(323)
plt.imshow(img2)
plt.title("Hough transform")
plt.show()




