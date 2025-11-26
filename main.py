import cv2 as cv
import numpy as np
import os
import csv

# ================= Video Input =================
folder = ""  # PATH
def inputVideo(path): # Generates grayscale frames from video
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        print(f"Failed to open video: {path}")
        exit()
        return None
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        yield gray
    capture.release() 
    
# ================= Optical Flow Calculation =================
def computeOpticalFlow(frame1, frame2): # Returns mag and ang of optical flow between two frames
    
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang 

# ================= Difference of Optical Flow Calculation (D3) =================
def DifferenceofOpticalFlow(mag1, ang1, mag2, ang2):
    magAccel = np.abs(mag2 - mag1) 
    rawAng = np.abs(ang2 - ang1)
    
    wrapAng = (2 * np.pi) - rawAng
    angAccel = np.minimum(rawAng, wrapAng)    
    
    alpha = 0.7
    beta  = 0.3
    combinedAccel = alpha * magAccel + beta * angAccel
    return combinedAccel # Numpy array of combined acceleration values

# ================= CSV File Writing =================
def saveToCsv(combinedAccel, videoName, outputFolder="csv"):
    os.makedirs(outputFolder, exist_ok=True)
    filename = "Real Video Data.csv"
    filepath = os.path.join(outputFolder, filename)
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Video_Name", "Combined_Accel"])  # header
            writer.writerow([videoName, combinedAccel])
        print(f"CSV file created at {filepath} with video {videoName}")
    else:
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([videoName, combinedAccel])
        print(f"Appended data for {videoName} to existing CSV at {filepath}")

    print(f"Saved CSV for {videoName} → {filepath}")

# ================ Classifier using Threshold =================
# Still working on finding the threshold through personal experimentation.
# Just worked through the data collection phase, and although this is a training-less model
# it still needs the threshold value to classify between real and deepfake videos.

# Will be implementing this module in the future.

# ================= Main Execution Module =================
for file in os.listdir(folder):
    if not file.endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    path = os.path.join(folder, file)
    print(f"Processing: {file}")

    prev2 = None 
    prev1 = None   

    for frame in inputVideo(path):  
        if prev2 is not None and prev1 is not None:
            mag1, ang1 = computeOpticalFlow(prev2, prev1)
            mag2, ang2 = computeOpticalFlow(prev1, frame)
            accel = DifferenceofOpticalFlow(mag1, ang1, mag2, ang2)
            saveToCsv(np.mean(accel), file)

        prev2 = prev1
        prev1 = frame

    
    
    
    