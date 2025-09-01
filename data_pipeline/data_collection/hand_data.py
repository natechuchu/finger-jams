import pandas as pd
import numpy as np
import mediapipe as mp
import os

class HandData:
    '''
    This class provides functions for data_collection.py to collect data from the live stream
    '''
    def __init__(self):
        self.data= []
        self.capture_count = 0
        self.label = None 
        self.gesture_to_index = {
            'pos_0': 0, # Other
            'pos_1': 1,
            'pos_2': 2,
            'pos_3': 3,
            'pos_4': 4,
            'pos_5': 5,
            'pos_6': 6,
            'pos_7': 7
        }
    
    def set_gesture_label(self, label: str):
        # Function to check if a label is valid or not
        if label not in self.gesture_to_index:
            print("ERROR: INVALID GESTURE LABEL!")
            return False
        else:
            print("Label sucessfully set.")
            self.label = label
            return True

    def add_coord(self, results): # coords = list[tuple[float, float, float]]
        # Function to save coord

        if not self.label: # If there is no label
            print("ERROR: label must be set first!")
        elif results.multi_hand_landmarks is None: # If hands were not detected
            print("ERROR: No hands were detected!")
        elif len(results.multi_hand_landmarks) == 2: # If two hands were detected 
            print("ERROR: Two hands are detected. Cannot save coords.")
        elif results.multi_handedness[0].classification[0].label == 'Left': # If left hand was detected
            print("ERROR: Left hand was detected. Cannot save coords.")
        else: 
            hand_landmarks = results.multi_hand_landmarks[0] # Get first coords from first hand
            coordinates = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark] # Get x, y, z coordinates for each 21 landmark objects
            self.data.append(coordinates) # Save the coords
            self.capture_count += 1 # Increase the capture count
            print(f"Added coordinate set #{self.capture_count} for label {self.label}")  # Print success statement
          
    def delete_last_coord(self):
        # Function to delete the previous coordinates saved
        if self.data:
            self.data.pop()
            print(f"Capture #{self.capture_count} sucessfully deleted")
            self.capture_count -=1
        else:
            print("ERROR: No captures to delete") 
          
    def get_unique_filename(self, base_name: str):
        # Function to get a unique file name
        i = 1
        name = base_name
        while os.path.exists(name):
            name = f"{base_name.rstrip('.npz')}_{i}.npz"
            i += 1
        return name    
    
    def save_data(self, filename: str):
        # Function to store coords in a numpy array with a unique file name

        # Error checking
        if not self.data: 
            print("ERROR: No data to save!")
            return
        if not self.label:
            print("ERROR: No label!")
            return
        
        # Store coords and labels as numpy array
        coords_array = np.array(self.data, dtype=np.float32)
        labels_array = np.array([self.gesture_to_index[self.label]] * self.capture_count)

        # Save to unique file name
        filename = self.get_unique_filename(filename)
        np.savez(f"{filename}", coords=coords_array, labels=labels_array)
        print(f"Data saved to {filename}") 
        
