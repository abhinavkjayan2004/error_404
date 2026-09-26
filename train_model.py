import torch
import torch.nn as nn
import numpy as np

class IMUDeadReckoningLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=2):
        super(IMUDeadReckoningLSTM, self).__init__()
        # Simplified encoder without LayerNorm to ensure clean ONNX graph translation
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU()
        )
        # Recurrent temporal backbone (LSTM)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Decoder layer mapping the final timestep to 2D displacement (delta_x, delta_y)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out = self.encoder(x)
        out, (hn, cn) = self.lstm(out)
        final_timestep_state = out[:, -1, :]
        displacement_pred = self.decoder(final_timestep_state)
        return displacement_pred

if __name__ == "__main__":
    model = IMUDeadReckoningLSTM()
    dummy_input = torch.randn(32, 50, 9)
    predictions = model(dummy_input)
    print("Model initialized successfully! Output shape:", predictions.shape)