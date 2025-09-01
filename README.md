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

### Processing

### Feature Engineering


## Modeling

## 🧪 Example Usage

## ⏳ Future Work
