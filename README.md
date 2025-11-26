# AI Video Distinguisher


## Purpose
To distinguish synthetic videos from authentic videos by analyzing temporal inconsistencies.

Heavily references methodology used in this paper:
[D3: Training-Free AI-Generated Video Detection Using Second-Order Features](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_D3_Training-Free_AI-Generated_Video_Detection_Using_Second-Order_Features_ICCV_2025_paper.html)

Although only used the simplified version of their operations, or at least an attempt.

Here I used 2 datasets, each corresponding for synthetic and authentic videos. The datasets were used to get the proper thresholds.
- **Synthetic videos**: Personally made dataset
- **Authentic videos**: [HMDB51 Human Model Database](https://serre.lab.brown.edu/#/resources), HMDB51: Human Model Database, swing_baseball folder 

## Warning
Do not use this on long clips. Maximum recommended input clip length: **10 seconds**, pending testing.

## How it works
The code extracts grayscaled frames from a video, computes dense optical flow using OpenCV's Farneback method and calculates the difference of consecutive flows thereby obtaining motion acceleration. 

Currently not done, as I am still trying to find the threshold. Although, in its complete state it should gather all of the motion acceleration (averaged) of all frames for a video, obtain its standard deviation and compare it to a set threshold.

This is a training-less model.

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`


## Usage

Not yet completed, as the program isn't completed yet.