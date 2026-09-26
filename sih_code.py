import os
import glob
import pandas as pd
import numpy as np

def load_and_sync_nclt_session(session_dir):
    """
    Parses ms25.csv (IMU) and groundtruth.csv from an NCLT session folder,
    performs nearest-neighbor timestamp matching, and creates 50-sample sliding windows.
    """
    # Define file paths
    imu_path = os.path.join(session_dir, "ms25.csv")
    gt_path = os.path.join(session_dir, "groundtruth.csv")
    
    if not os.path.exists(imu_path) or not os.path.exists(gt_path):
        raise FileNotFoundError(f"Required ms25.csv or groundtruth.csv missing in {session_dir}")

    print(f"Loading data from {session_dir}...")
    
    # NCLT CSV files usually do not have headers; typical column structure:
    # ms25.csv: [timestamp, m_x, m_y, m_z, a_x, a_y, a_z, g_x, g_y, g_z] (or similar microsecond timestamps)
    # groundtruth.csv: [timestamp, x, y, z, roll, pitch, yaw, ...]
    imu_df = pd.read_csv(imu_path, header=None)
    gt_df = pd.read_csv(gt_path, header=None)
    
    # Rename primary columns for convenience (assuming column 0 is timestamp)
    imu_df.rename(columns={0: 'timestamp'}, inplace=True)
    gt_df.rename(columns={0: 'timestamp'}, inplace=True)
    
    # Sort by timestamps to enable merge_asof
    imu_df = imu_df.sort_values('timestamp').reset_index(drop=True)
    gt_df = gt_df.sort_values('timestamp').reset_index(drop=True)
    
    print("Synchronizing IMU streams with ground truth using nearest-neighbor matching...")
    # Perform nearest-neighbor timestamp matching (tolerance set to 50,000 microseconds / 50ms)
    synchronized_df = pd.merge_asof(
        imu_df, 
        gt_df, 
        on='timestamp', 
        direction='nearest', 
        tolerance=50000
    )
    
    # Drop rows where ground truth couldn't be matched within tolerance
    synchronized_df = synchronized_df.dropna().reset_index(drop=True)
    
    # Extract feature matrix (IMU readings) and target coordinates (x, y from ground truth)
    # Adjust column indices based on standard NCLT layouts (e.g., columns 1-9 for IMU features)
    imu_features = synchronized_df.iloc[:, 1:10].values # Magnetometer, Accelerometer, Gyroscope
    
    # Assuming x and y ground-truth coordinates are located at specific indices (e.g., columns corresponding to x and y)
    # Let's extract generic position coordinates columns if available, or specify indices explicitly
    # Typically in NCLT ground truth: col 1 is x, col 2 is y
    positions = synchronized_df.iloc[:, [1, 2]].values 

    print(f"Creating 50-sample temporal sliding windows...")
    window_size = 50
    X_windows = []
    y_displacements = []
    
    for i in range(len(synchronized_df) - window_size):
        # Window of 50 consecutive IMU samples (1 second at 50Hz)
        window_data = imu_features[i : i + window_size]
        
        # Target: displacement vector from start of window to end of window, or step-wise delta
        p_start = positions[i + window_size - 1]
        p_end = positions[i + window_size]
        delta_xy = p_end - p_start
        
        X_windows.append(window_data)
        y_displacements.append(delta_xy)
        
    X_windows = np.array(X_windows)
    y_displacements = np.array(y_displacements)
    
    print(f"Generated {X_windows.shape[0]} windows with shape {X_windows.shape[1:]}")
    return X_windows, y_displacements

# Example execution template:
if __name__ == "__main__":
    # Point to your extracted NCLT session folder path
    sample_session_dir = "data/2013-01-10"
    if os.path.exists(sample_session_dir):
        X, y = load_and_sync_nclt_session(sample_session_dir)