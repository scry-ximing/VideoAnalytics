# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:57:32 2019

@author: ximing

This script take the image and the result saved by Mask-RCNN script in google colab as inputs
The script will give the median color for each detected person. 
The color is saved as a json file. The key for each entry is the bounding box of each person, 
the value is the median of all colors, whcih can be used as an feature for classification. 


"""
import argparse
import cv2
import numpy as np
import json


def colorExtractor(impath,mask):
    """
    color extractor for one image and one mask
    """
    img = cv2.imread(impath)
    masked = img[mask]    
    return [np.median(masked[:,0]),np.median(masked[:,1]),np.median(masked[:,2])]


def colorExtractorAll(impath,resultsPath):
    """
    color extractor for one image all masks in that image
    """
    r = np.load(resultsPath).item()
    masks = r['masks']
    masks = np.moveaxis(masks,2,0)
    people={}
    for mask, cid,roi in zip(masks,r['class_ids'],r['rois']):
        if cid==1:
            people[str(roi)]=colorExtractor(impath,mask)
    return people

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Color Extractor')
    parser.add_argument('--i',  
                    help='image path')
    parser.add_argument('--r', 
                        help='results file')# results file are saved from the Mask-RCNN script in google colab. 
    
    args = parser.parse_args()
    impath=args.i
    resultsPath = args.r
    outFile = impath+'_color.json'
    colors=colorExtractorAll(impath,resultsPath)
    with open(outFile, 'w') as fp:
        json.dump(colors, fp)
