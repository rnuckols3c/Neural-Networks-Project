import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BidirectionalAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layers
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size*2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        out = self.fc(context_vector)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        
        return out, attention_weights

class RegimeIdentificationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_regimes, dropout=0.2):
        super(RegimeIdentificationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature extraction with LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Regime classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_regimes)
        )
        
        # Probability output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Process with LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use the last hidden state from both directions
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Apply layer normalization
        h_n = self.layer_norm(h_n)
        
        # Regime classification
        logits = self.classifier(h_n)
        probabilities = self.softmax(logits)
        
        return logits, probabilities
