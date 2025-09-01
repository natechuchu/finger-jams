from data_pipeline.data_collection.hand_tracker import HandTracker
import cv2
from data_pipeline.normalize_utils import normalize_psar, normalize_lpsar, compute_thumb_angle
import tensorflow as tf
import pandas as pd
import numpy as np
import fluidsynth
import os
os.environ['FLUIDSYNTH_LIB_PATH'] = r'C:\Users\natec\Documents\Coding\finger-jams\audio_tools\fluidsynth-2.4.7-win10-x64\bin\libfluidsynth-3.dll'
import fluidsynth

model = tf.keras.models.load_model('./models/hand_gesture_model_9.keras')
note_mapping = {
    0: None,
    1: 12,  # C0
    2: 14,  # D0
    3: 16,  # E0
    4: 17,  # F0
    5: 19,  # G0
    6: 21,  # A0
    7: 23   # B0
}

def get_prediction(hand_data):
   
    results = []
    # Error checking
    if not hand_data.multi_hand_landmarks:
        return results
    # Extract coordinates
    for idx, hand in enumerate(hand_data.multi_hand_landmarks):
        coordinates = [(lm.x, lm.y, lm.z) for lm in hand.landmark] # Get x, y, z coordinates for each 21 landmark objects

        # Convert to numpy array
        coordinates = np.array(coordinates, dtype=np.float64)

        # Get Label
        label = hand_data.multi_handedness[idx].classification[0].label

        # Normalize, convert to numpy, reshape
        if label == "Right":
            data = normalize_psar(coordinates)

        else:
            data = normalize_lpsar(coordinates)

        sin, cos = compute_thumb_angle(data)

        data = data.reshape(1,-1)
        angles = np.array([sin, cos], dtype=np.float64).reshape(1,2)
        data = np.concatenate([data, angles], axis=1)


        # Make predictions and print
        predictions = model.predict_on_batch(data)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        results.append([label, predicted_class, confidence])

    return results

def get_note(results): # Return the note to play
    right_class = 0
    left_class = 0
    right_conf = 0
    left_conf = 0
    note = None 

    for hand in results:
        if hand[0] == "Right":
            right_class = hand[1]
            right_conf = hand[2]
        elif hand[0]== "Left":
            left_class = hand[1]
            left_conf = hand[2]
    if left_class != 0 and right_class != 0:
        octave = left_class + 2 # Class 1 will start at MIDI octave 3
        note = note_mapping.get(right_class) + (octave * 12)
    results_dict = {
        'note': note,
        'right_class': right_class,
        'left_class' : left_class,
        'right_conf' : right_conf,
        'left_conf' : left_conf
    }
    return results_dict
    
def main():
    # Declarations
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    fs = fluidsynth.Synth('./audio_tools/FluidR3_GM.sf2')
    frame_count = 0
    prev_note = None
    results_dict = None
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Synth settings
    fs.start()
    sfid = fs.sfload("./audio_tools/FluidR3_GM.sf2") 
    fs.program_select(0, sfid, 0, 0)

    # Livestream
    while cap.isOpened():
        # Collect frames
        success, image = cap.read()
        frame_count += 1
        if not success:
            print("Ignoring empty camera frame.")
            continue
    
        # Mirror image
        image = cv2.flip(image, 1)
        
        # Get mediapipe results
        image.flags.writeable = False
        hand_data = tracker.process(image)
        image.flags.writeable = True

        # Get results every 5 frames
        if frame_count % 5 == 0:
            results = get_prediction(hand_data)
            results_dict = get_note(results)

            # Set current_note
            current_note = results_dict['note']

            # Only if a change happens
            if current_note != prev_note: 
                if prev_note: # Turn off previous note if on
                    fs.noteoff(0, prev_note)
                if current_note: # If the current note is not none, play the note
                    fs.noteon(0, current_note, 127)
                # Set previous note 
                prev_note = current_note 
        
        

        # Draw landmarks
        annotated_image = tracker.draw_hands(image, hand_data)


        # Draw predictions
        if results_dict:
            cv2.putText(annotated_image, f"Left: {results_dict['left_class']}, {round(results_dict['left_conf'] * 100, 2)}% | Right: {results_dict['right_class']}, {round(results_dict['right_conf'] * 100, 2)}%"
                    , (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
        # Display image      
        cv2.imshow('MediaPipe Hands', annotated_image)

        # Collect user input key
        key = cv2.waitKey(5) & 0xFF

        if key == 27: # ESC to quit
            print("Camera is closing...")
            break
    if current_note:
        fs.noteoff(0, current_note)
    cap.release() 
    cv2.destroyAllWindows


if __name__ == "__main__":
    main()     

     