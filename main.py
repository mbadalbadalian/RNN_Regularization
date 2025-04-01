import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
params = {
    "batch_size": 20,
    "seq_length": 20,
    "layers": 2,
    "decay": 2,
    "rnn_size": 200,
    "dropout": 0.0,
    "init_weight": 0.1,
    "lr": 1.0,
    "vocab_size": 10000,
    "max_epoch": 4,
    "max_max_epoch": 13,
    "max_grad_norm": 5,
}

# Load PTB dataset (replace with actual dataset loading)
def load_ptb_data():
    data = torch.randint(0, params["vocab_size"], (100000,)).long()
    return data.to(device)

data_train = load_ptb_data()
data_valid = load_ptb_data()
data_test = load_ptb_data()

# LSTM Cell implementation
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.W = nn.Linear(input_size, 4*hidden_size)
        self.U = nn.Linear(hidden_size, 4*hidden_size)
    
    def forward(self, x, prev_c, prev_h):
        gates = self.W(x) + self.U(prev_h)
        i, f, o, g = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_c, next_h

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, params):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(params["vocab_size"], params["rnn_size"])
        self.lstm_cells = nn.ModuleList([
            LSTMCell(params["rnn_size"], params["rnn_size"]) for _ in range(params["layers"])
        ])
        self.fc = nn.Linear(params["rnn_size"], params["vocab_size"])
        self.dropout = nn.Dropout(params["dropout"])
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-params["init_weight"], params["init_weight"])

    def forward(self, x, hidden_states):
        x = self.embedding(x)
        next_hidden_states = []
        for i, cell in enumerate(self.lstm_cells):
            prev_c, prev_h = hidden_states[i]
            next_c, next_h = cell(x, prev_c, prev_h)
            x = next_h  # Input for next layer
            next_hidden_states.append((next_c, next_h))
        x = self.dropout(x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1), next_hidden_states

    def init_hidden(self, batch_size):
        return [(torch.zeros(batch_size, params["rnn_size"]).to(device),
                 torch.zeros(batch_size, params["rnn_size"]).to(device))
                for _ in range(params["layers"])]

# Initialize model
model = LSTMModel(params).to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"])

def fp(data, hidden_states):
    model.train()
    x = data[:-1]
    y = data[1:]
    x, y = x.to(device), y.to(device)
    output, hidden_states = model(x, hidden_states)
    loss = criterion(output.view(-1, params["vocab_size"]), y.view(-1))
    return loss, hidden_states

def bp(loss):
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
    optimizer.step()
    torch.cuda.synchronize()
    import gc
    gc.collect()

def run_epoch(data):
    hidden_states = model.init_hidden(params["batch_size"])
    total_loss = 0
    num_batches = max(1, data.size(0) // params["seq_length"] - 1)
    for batch in range(num_batches):
        x = data[batch * params["seq_length"]:(batch + 1) * params["seq_length"]]
        y = data[batch * params["seq_length"] + 1:(batch + 1) * params["seq_length"] + 1]
        loss, hidden_states = fp(x, hidden_states)
        bp(loss)
        total_loss += loss.item()
        hidden_states = [(c.detach(), h.detach()) for (c, h) in hidden_states]
    return torch.exp(torch.tensor(total_loss / num_batches))

def evaluate(data):
    model.eval()
    hidden_states = model.init_hidden(params["batch_size"])
    total_loss = 0
    num_batches = max(1, data.size(0) // params["seq_length"] - 1)
    with torch.no_grad():
        for batch in range(num_batches):
            x = data[batch * params["seq_length"]:(batch + 1) * params["seq_length"]]
            y = data[batch * params["seq_length"] + 1:(batch + 1) * params["seq_length"] + 1]
            loss, hidden_states = fp(x, hidden_states)
            total_loss += loss.item()
            hidden_states = [(c.detach(), h.detach()) for (c, h) in hidden_states]
    return torch.exp(torch.tensor(total_loss / num_batches))

# Training loop
for epoch in range(params["max_max_epoch"]):
    train_ppl = run_epoch(data_train)
    valid_ppl = evaluate(data_valid)
    print(f"Epoch {epoch+1}, Train PPL: {train_ppl:.2f}, Valid PPL: {valid_ppl:.2f}")
    if epoch > params["max_epoch"]:
        params["lr"] /= params["decay"]

# Test evaluation
test_ppl = evaluate(data_test)
print(f"Test PPL: {test_ppl:.2f}")