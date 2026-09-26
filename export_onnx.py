import torch
from train_model import IMUDeadReckoningLSTM

def export_to_onnx():
    device = torch.device("cpu")
    model = IMUDeadReckoningLSTM(input_size=9, hidden_size=64, output_size=2).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load("imu_lstm_model.pth", map_location=device))
    model.eval()
    
    dummy_input = torch.randn(1, 50, 9, device=device)
    onnx_file_path = "imu_lstm_model.onnx"
    
    # Use legacy exporter (dynamo=False) to ensure clean LSTM node translation
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_windows'],
        output_names=['predicted_displacement'],
        dynamic_axes={
            'input_windows': {0: 'batch_size'},
            'predicted_displacement': {0: 'batch_size'}
        },
        dynamo=False
    )
    print(f"Model successfully exported to {onnx_file_path}")

if __name__ == "__main__":
    export_to_onnx()