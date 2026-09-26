import os
import numpy as np
import torch
import onnxruntime as ort
from sih_code import load_and_sync_nclt_session

def evaluate_dead_reckoning():
    # 1. Load a validation or test session (e.g., using the same or another session)
    session_dir = "data/2013-01-10"
    X, y_true_displacements = load_and_sync_nclt_session(session_dir)
    
    # 2. Initialize ONNX Runtime session for inference
    onnx_path = "imu_lstm_model.onnx"
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_name = ort_session.get_inputs()[0].name
    
    # 3. Run sequential free-run integration
    print("Running free-run trajectory integration...")
    predicted_displacements = []
    
    # Batch inference for efficiency
    for i in range(0, len(X), 64):
        batch_x = X[i:i+64].astype(np.float32)
        ort_inputs = {input_name: batch_x}
        preds = ort_session.run(None, ort_inputs)[0]
        predicted_displacements.append(preds)
        
    predicted_displacements = np.vstack(predicted_displacements)
    
    # Reconstruct trajectory by cumulative sum of predicted displacements
    reconstructed_path = np.cumsum(predicted_displacements, axis=0)
    true_path = np.cumsum(y_true_displacements, axis=0)
    
    # Calculate Endpoint Error / RMSE
    rmse = np.sqrt(np.mean((reconstructed_path - true_path) ** 2))
    print(f"Trajectory Reconstruction Evaluation Complete!")
    print(f"Position RMSE against Ground Truth: {rmse:.4f} meters")

if __name__ == "__main__":
    evaluate_dead_reconng = evaluate_dead_reckoning()