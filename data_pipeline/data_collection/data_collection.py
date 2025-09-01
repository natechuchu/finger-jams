from hand_tracker import HandTracker
from hand_data import HandData
import cv2

GESTURE_LABEL = "pos_0"  
FILE_PATH = "C:/Users/natec/Documents/Coding/finger-jams"

def print_menu():
    print("MENU:")

    print("Save coordinates    - Enter")
    print("Delete last coords  - Backspace")

def main():
    # Declarations
    tracker = HandTracker()
    data = HandData()
    cap = cv2.VideoCapture(0)

    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Set label
    if not data.set_gesture_label(GESTURE_LABEL):
        return
    print_menu()

    # Livestream
    while cap.isOpened():
        # Collect frames
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the image
        image = cv2.flip(image, 1)

        # Get mediapipe results
        image.flags.writeable = False
        results = tracker.process(image)
        image.flags.writeable = True

        # Draw landmarks
        annotated_image = tracker.draw_hands(image, results)
        
        # Display image      
        cv2.imshow('MediaPipe Hands', annotated_image)

        # Collect user input key
        key = cv2.waitKey(5) & 0xFF

        if key == 13: # Enter to save coord
            data.add_coord(results)
        elif key == 8: # Backspace to delete last coord
            data.delete_last_coord()
        elif key == 32:
            data.print_predictions(results)
        elif key == 27: # ESC to quit
            print("Camera is closing...")
            break
    cap.release() 
    cv2.destroyAllWindows

    # Give user choice to save data
    while True:
        ans = input(f"Would you like to save this data to {GESTURE_LABEL}_data? (y/n): ")
        if ans in ('y', 'n'):
            break
        print("Please enter valid input (y/n).")

    if ans == 'y':

        data.save_data(f"{FILE_PATH}/datasets/raw/{GESTURE_LABEL}_data.npz")
    else:
        print("Data is not saved. Goodbye...")


if __name__ == "__main__":
    main()     

     