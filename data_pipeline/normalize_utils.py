import numpy as np
import pandas as pd
import os


def normalize_position(sample: np.ndarray):
    # Set origin as the wrist
    origin = sample[0]

    # Center landmarks around wrist
    return sample - origin

def normalize_scale(sample: np.ndarray):
    # Get the middle finger base coordinate
    mid_finger_base = sample[9]

    # Get the norm of the vector from the wrist
    scale = np.linalg.norm(mid_finger_base)

    # Scale the sample 
    return sample/scale

def normalize_alignment(sample: np.ndarray):
    # Get vector from wrist to middle_finger_base
    v = sample[9] - sample[0]

    # Get vector where hand should be aligned
    target = np.array([0, 1, 0])

    # Get normalized axis of rotation between v and target
    rot_axis = np.cross(v, target)
    rot_axis = rot_axis/np.linalg.norm(rot_axis)

    # Get angle between v and target
    v_len = np.linalg.norm(v)
    target_len = np.linalg.norm(target)
    angle = np.arccos(np.dot(v, target) / (v_len * target_len))

    #Rodrigues rotation matrix
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    I = np.identity(3)
    rot_matrix = I + np.sin(angle) * K + (1-np.cos(angle)) * (K @ K)

    # Get the rotated landmarks
    rot_lms = sample @ rot_matrix.T
    
    return rot_lms

def normalize_rotation(sample: np.ndarray):
    # Get vector from wrist to thumb base 
    v = sample[1] - sample[0]

    # Get the projection of the vector on the x-z plane
    v_proj = np.array([v[0], 0, v[2]])

    # Get the angle between the projection and the x-axis
    theta = np.arctan2(v_proj[2], v_proj[0])

    # Form angle rotation matrix
    rot_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Apply rotation matrix and return
    rotated = sample @ rot_matrix.T
    return rotated

def normalize_psar(sample: np.ndarray):
    sample = normalize_position(sample)
    sample = normalize_scale(sample)
    sample = normalize_alignment(sample)
    sample = normalize_rotation(sample)
    return sample

def normalize_left(left: np.ndarray):
    # Form matrix to flip array across x-axis
    flip_x = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # Apply matrix and return
    right = left @ flip_x.T
    return right

def normalize_lpsar(sample: np.ndarray):
    sample = normalize_left(sample)
    sample = normalize_position(sample)
    sample = normalize_scale(sample)
    sample = normalize_alignment(sample)
    sample = normalize_rotation(sample)
    return sample

def compute_thumb_angle(sample: np.ndarray):
    # Get vectors
    v_thumb  = sample[4] - sample[1]
    v_palm = sample[9]- sample[0]

    # Get norms
    norm_thumb = np.linalg.norm(v_thumb)
    norm_palm = np.linalg.norm(v_palm)

    # Get denominator and return if 0
    denom = norm_thumb * norm_palm
    if denom == 0:
        return 0, 1
    
    # Calculate the angle between the two vectors
    angle = np.arccos(np.clip(np.dot(v_thumb, v_palm) / denom, -1.0, 1.0))
    
    # Return the sin and cosine of the angles
    return (np.sin(angle), np.cos(angle))


def save_to_file(coords: np.ndarray, angles: np.ndarray, labels: np.ndarray,  file_name: str):
    i = 1
    original_name = "./Data/" + file_name.rstrip('.npz') + '_processed.npz'
    name = original_name
    while os.path.exists(name):
        name = f"{original_name.rstrip('.npz')}_{i}.npz"
        i += 1

    np.savez(name, coords=coords, angles=angles, labels=labels)
    print(f"Data saved to {name}")


def main():
    # Load Data
    file_name = "raw/pos_7_data_3.npz"
    data = np.load( f"./Data/{file_name}")
    coords = data['coords']
    labels = data['labels']

    angles = []
    normalized_data = []

    # For each sample in the file
    for sample in coords:
        
        # Normalize data
        norm_sample = normalize_psar(sample)
        
        # Get and store angles
        angle = compute_thumb_angle(norm_sample)
        
        # Store data
        angles.append(angle)
        normalized_data.append(norm_sample)

    # Convert data to numpy arrays
    angles = np.stack(angles)
    normalized_data = np.stack(normalized_data)

    # Save to file
    save_to_file(normalized_data, angles, labels, file_name)

if __name__ == "__main__":
    main()  