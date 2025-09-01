# Virtual Instrument using Mediapipe and NN-Classifier

This project utilizes a neural network classifier to classify 7 hand gestures from Mediapipe and OpenCV inputs. Each combination of hand gestures and handness is mapped to one of 49 notes in the C major scale.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Background](#background)
- [Features](#features)
- [Example Usage](#example-usage)
- [Future Work](#future-work)


---

## 🧠 Overview
Using Mediapipe as a hand recognizer and OpenCV for camera input, I manually collected over 10,000 hand gesture samples using my HandData class. From there, I built a custom pipeline to preprocess the data,
engineered additional features to distinguish similar gestures, and trained a neural network classifier. Using the left hand to control the octaves, and the right hand to control the 7 notes in a C Major scale, I connected my preprocessing pipeline
to a live input and mapped the processed data to audio outputs. In the end, I created a novel design for next-generation music creation!

---

## 🖼️ Background
Ever since I started playing violin at 5 years old, I've always loved creating and practicing music. Whether it's practicing the Bruch Concerto in G minor for the thousandth time or jazzifying Star Wars music off the top of my head,
music is where we can explore how deep our creativity can go. I believe that everyone should have access to create music, but often it can be inconvenient and pricey to own an instrument. 
So I decided to create a virtual instrument - an instrument that can be played anywhere, anytime, completely hassle-free! All it requires is a free set of hands and your imagination. 

## ✨ Features

## Data Pipeline
### Collection
Data collection takes place in `data_collection.py`. This script utilized two classes: HandTracker and HandData
- HandTracker
  - This class provided the necessary methods to process an image input, collect the hand landmark coordinates, and draw the landmarks on the image
- HandData
  - This class provided the necessary tools to collect and save the coordinates. Utilities included during a capture session include setting the gesture label for a session, adding/deleting coordinates, and saving the data to a unique file name
- How to use `data_collection.py`
  1) Before running the script, ensure you have set the proper "GESTURE_LABEL" and "FILE_PATH"
  2) OpenCV will launch on start and will open up another tab with the live video feed
  3) Hit Enter to save a coordinate for the right hand
     - Hand must be fully in the screen to register
     - The left hand cannot be captured
     - Two hands cannot be captured
  4) Hit Backspace to delete the last captured coordinate
  5) Hit ESC to exit the program.
  
### Processing
Data processing takes place with the `normalize_utils.py` script. See `edda.ipynb` for my full reasoning for each processing step.
1) Raw data
Lets take a look at one sample
### Feature Engineering


## Modeling

## 🧪 Example Usage

## ⏳ Future Work
