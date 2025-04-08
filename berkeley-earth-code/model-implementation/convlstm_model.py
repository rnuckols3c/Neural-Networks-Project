import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # for the 4 gates
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, input_tensor, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class UHIAnalysisModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels):
        super(UHIAnalysisModel, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        # ConvLSTM layers
        self.conv_lstm_cells = nn.ModuleList()
        self.conv_lstm_cells.append(ConvLSTMCell(input_channels, hidden_channels[0], kernel_size))
        
        for i in range(1, num_layers):
            self.conv_lstm_cells.append(
                ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size)
            )
        
        # Attention mechanism for spatial weighting
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output convolutional layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
        
        # Additional feature extraction for urban areas
        self.urban_classifier = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, urban_mask=None):
        # x shape: [batch_size, time_steps, channels, height, width]
        batch_size, time_steps, _, height, width = x.size()
        
        # Initialize hidden states for each layer
        h_states = []
        c_states = []
        
        for i in range(self.num_layers):
            h_states.append(torch.zeros(batch_size, self.hidden_channels[i], height, width).to(x.device))
            c_states.append(torch.zeros(batch_size, self.hidden_channels[i], height, width).to(x.device))
        
        # Process each time step
        output_sequence = []
        
        for t in range(time_steps):
            input_frame = x[:, t]
            
            # Pass through ConvLSTM layers
            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    h_next, c_next = self.conv_lstm_cells[layer_idx](
                        input_frame, (h_states[layer_idx], c_states[layer_idx])
                    )
                else:
                    h_next, c_next = self.conv_lstm_cells[layer_idx](
                        h_states[layer_idx-1], (h_states[layer_idx], c_states[layer_idx])
                    )
                
                h_states[layer_idx] = h_next
                c_states[layer_idx] = c_next
            
            # Get current output
            current_output = self.output_conv(h_states[-1])
            output_sequence.append(current_output)
        
        # Stack outputs along time dimension
        outputs = torch.stack(output_sequence, dim=1)
        
        # Apply spatial attention to final hidden state
        attention_weights = self.spatial_attention(h_states[-1])
        attended_features = h_states[-1] * attention_weights
        
        # Urban heat island identification
        urban_probability = self.urban_classifier(attended_features)
        
        # If urban mask is provided, use it to focus only on urban areas
        if urban_mask is not None:
            urban_probability = urban_probability * urban_mask
        
        return outputs, urban_probability, attention_weights
