# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:12:46 2019

@author: ximing
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter


def distance(boxA,boxB):
    """
    Calculate the distance between the center of two bounding boxes
    """
    xAmid = (boxA[0]+boxA[2])/2
    xBmid = (boxB[0]+boxB[2])/2
    return abs(xAmid-xBmid)

def interboxB(boxA, boxB):
	"""
  Determine ratio of the overlapped area in the second bounding box
  inputs are two bounding box coordinates. (x1,y1,x2,y2). (x1,y1) is the left top coordinates of the bounding box
  (x2,y2) is the right bottom coordinates of the bounding box
  output is the ratio
  
  """
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	boxBratio = interArea / boxBArea
 
	# return the intersection over union value
	return boxBratio

def showTime(df,pid,fps):
    """
    input is the dataframe of the tracking data, person ID, and frames per second
    return this person(based on the person ID)'s first appearance time, and last appearance time
    """
    frames = df[df['id']==pid].frame # select all related frames
    return [min(frames)*(1/fps),max(frames)*(1/fps)] # return first frame and last frame, convert to time

def isWorking(working_area,box):
    """
    Determine if the employee is at the working area. 
    If the person's bounding box's 50% is in the working area, this person is identified as working. 
    inputs are working-area's bounding box and the person's bounding box
    """
    boxBratio = interboxB(working_area,box)
    if boxBratio>.5:
        return True
    return False

def isWorkingPair(boxA,boxB):
    """
    Determine if the employee is working with another person(can be customer or employee). 
    If two people's bounding box's distance is less than 100 pixel, the employee is determined as working. 
    """
    if distance(boxA,boxB)<100:
        return True
    return False

def isPairWorkingFrame(df,frame,pid):
    """
    Determine if the person with the pid is working with other people in this frame
    """
    df2 = df[df['frame']==frame]
    if not len(df2) or len(df2)<2 or pid not in set(df2['id']):# check if the person with the pid is in this frame
        return False
    boxes={}
    for i in range(len(df2)):
        # find all boxes in the frame
        boxes[df2.iloc[i]['id']]=[df2.iloc[i]['x'],df2.iloc[i]['y'],df2.iloc[i]['x']+df2.iloc[i]['dx'],df2.iloc[i]['y']+df2.iloc[i]['dy']]
    for pid2 in boxes:# find if anybody is pair working with the employee
        if pid==pid2:
            continue
        if distance(boxes[pid],boxes[pid2]):
            return True
    return False

def isWorkingFrame(working_area,frame,pid):
    """
    Determine if the person with the pid is working(in the working area) in this frame
    """
    df2 = df[(df['id']==pid) & (df['frame']==frame) ]
    if not len(df2):
        return "Not Identified"
    box=[df2.iloc[0]['x'],df2.iloc[0]['y'],df2.iloc[0]['x']+df2.iloc[0]['dx'],df2.iloc[0]['y']+df2.iloc[0]['dy']]
    if isWorking(working_area,box):
        return 'Working'
    if isPairWorkingFrame(df,frame,pid):
        return 'Working'
    else:
        return 'Not Working'
    

    
def isWorkingAll(working_area,df,pid):
    """
    inputs are working area bounding box, tracking data and person's id
    outputs are all the frames in which the person is working, and all frames in which the person shows 
    """
    df2=df[df['id']==pid]
    working_frames=[]
    show_frames=[]
    for i in range(len(df2)):
        box=[df2.iloc[i]['x'],df2.iloc[i]['y'],df2.iloc[i]['x']+df2.iloc[i]['dx'],df2.iloc[i]['y']+df2.iloc[i]['dy']]
        show_frames.append(df2.iloc[i]['frame'])
        if isWorking(working_area,box) or isPairWorkingFrame(df,frame,pid):
            working_frames.append(df2.iloc[i]['frame'])
    return [working_frames,show_frames]



def workingTimeEachPerson(df,fps):
    """
    Print each person's total working time, work duration and show duration to the screen
    """
    for pid in df.id.unique():
        dftmp = df[(df['id']==pid) & (df['status']=='Working')]
        dfshow = df[df['id']==pid]
        if fid2name(id2name,pid)=='Customer':
            continue        
        print('object '+str(pid)+':'+fid2name(id2name,pid)+' total working time:'+str(len(dftmp)*(1/fps))+" seconds")        
        print('work duration '+': '+str(dftmp.iloc[0]['frame']*(1/fps))+"-"+str(dftmp.iloc[-1]['frame']*(1/fps))+" seconds")
        print('show duration '+': '+str(dfshow.iloc[0]['frame']*(1/fps))+"-"+str(dftmp.iloc[-1]['frame']*(1/fps))+" seconds")

def workingTimeEachPerson2File(df,fps,fpath):
    """
    Print each person's total working time, work duration and show duration to file 
    """
    file = open(fpath,"w") 
    for pid in df.id.unique():
        dftmp = df[(df['id']==pid) & (df['status']=='Working')]
        dfshow = df[df['id']==pid]
        if fid2name(id2name,pid)=='Customer':
            continue
        file.write('object '+str(pid)+':'+fid2name(id2name,pid)+' total working time:'+str(len(dftmp)*(1/fps))+" seconds\n")
        file.write('work duration '+': '+str(dftmp.iloc[0]['frame']*(1/fps))+"-"+str(dftmp.iloc[-1]['frame']*(1/fps))+" seconds\n")
        file.write('show duration '+': '+str(dfshow.iloc[0]['frame']*(1/fps))+"-"+str(dftmp.iloc[-1]['frame']*(1/fps))+" seconds\n")
    file.close()


def searchPerson(facedf,df,i):
    """
    Based on the face recognition results from facedf, and the tracking data from df,
    determine the the person's name. 
    """
    box,frame = [facedf.iloc[i]['y1'],facedf.iloc[i]['x1'],facedf.iloc[i]['y2'],facedf.iloc[i]['x2']],facedf.iloc[i]['frame']
    df2 = df[df['frame']==frame]
    percent=[]
    for j in range(len(df2)):
        boxA=[df2.iloc[j]['x'],df2.iloc[j]['y'],df2.iloc[j]['x']+df2.iloc[j]['dx'],df2.iloc[j]['y']+df2.iloc[j]['dy']]
        percent.append(interboxB(boxA,box))
    if not percent or max(percent)==0:
        return -1
    return df2.iloc[np.argmax(percent)]['id']

def loadData(tracker_file,faceRecog_file):
    """
    load and preprocess the tracker csv, face Recognition csv, output the dataframe
    """
    df = pd.read_csv(tracker_file,names=['frame','id','y','x','dy','dx','a','b','c','d'])
    facedf = pd.read_csv(faceRecog_file,names=['file','detected','name','x1','y1','x2','y2'])
    facedf = facedf[facedf['detected']==1]
    facedf['frame'] = [int(s[11:14]) for s in facedf['file']]
    return [df,facedf]

def identifyStatus(df):
    """
    Identify the status of each person in the tracker dataframe
    """
    status=[]
    for i in range(len(df)):
        pid = df.loc[i,'id']
        frame = df.loc[i,'frame']
        if isWorkingFrame(working_area,frame,pid)=='Working':
            status.append("Working")
        else:
            status.append('Not Working')
    return status

def matchingFace(df,facedf):
    """
    Use face Recognition dataframe to identify each person's identity in the tracker dataframe
    """
    pids=[]
    for i in tqdm(range(len(facedf))):
        pids.append(searchPerson(facedf,df,i))    
    facedf['pid']=pids
    id2name={}
    threshold=0.1
    for pid in set(pids):
        names = facedf[facedf['pid']==pid].name
        a=Counter(names).most_common(2)
        if len(a)<2:
            id2name[pid]=a[0][0]
        elif a[0][0]!='Customer':
            id2name[pid]=a[0][0]
        elif a[1][1]/len(names)>threshold:
            id2name[pid]=a[1][0]
        else:
            id2name[pid]='Customer'
    return id2name

def fid2name(id2name,pid):
    """
    Normalize the name for each person's id
    """
    if pid in id2name:
        return id2name[pid]
    return 'Customer'

if __name__ == "__main__":
    """
    In this script, the tracker file path, faceRecognition file path and working area are predefined. 
    Please modify these inputs based on your needs. 
    """
    tracker_file = 'videoTracking.csv'
    working_area = [0,0,720,260]
    faceRecog='faceRecogVideo2.csv'
    
    [df,facedf]=loadData(tracker_file,faceRecog)
    # find the name of each identified person
    id2name = matchingFace(df,facedf)
    names = [fid2name(id2name,pid) for pid in df['id']]
    df['person_names']=names
    # find the status of each person in each frame
    status = identifyStatus(df)
    df['status']=status    
    # write the data with person's name, status and summary report into csv and txt file. 
    workingTimeEachPerson2File(df,10,'summary.txt')
    df.to_csv('dfwstatus.csv')