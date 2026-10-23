import json
import time
import torch
import torch.nn as nn

DEFAULT_VOCAB_WORDS = [
    "openshift",
    "kubernetes",
    "kubeflow",
    "pipeline",
    "pytorch",
    "container",
    "tensor",
    "artificialintelligence",
    "machinelearning",
    "automation",
]

class CharLSTM(nn.Module):
    """
    A lightweight Character-Level Language Model based on LSTM.
    Predicts the next character in a word given preceding characters.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_vocab(words=None):
    if words is None:
        words = DEFAULT_VOCAB_WORDS
    unique_chars = sorted(list(set("".join(words) + "$ ")))
    char2idx = {c: i for i, c in enumerate(unique_chars)}
    idx2char = {i: c for i, c in enumerate(unique_chars)}
    return char2idx, idx2char


def train_model(words=None, epochs: int = 150, lr: float = 0.02, device: str = "cpu"):
    if words is None:
        words = DEFAULT_VOCAB_WORDS
    char2idx, idx2char = build_vocab(words)
    vocab_size = len(char2idx)

    torch_device = torch.device(device)
    model = CharLSTM(vocab_size=vocab_size).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    start_time = time.time()
    final_loss = 0.0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for word in words:
            # End-of-sequence token '$'
            tokens = [char2idx[c] for c in word + "$"]
            inputs = torch.tensor(tokens[:-1], dtype=torch.long, device=torch_device).unsqueeze(0)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=torch_device).unsqueeze(0)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        final_loss = epoch_loss / len(words)

    training_time_ms = (time.time() - start_time) * 1000
    return model, char2idx, idx2char, final_loss, training_time_ms


def complete_word(model: CharLSTM, char2idx: dict, idx2char: dict, prefix: str, max_length: int = 25, device: str = "cpu") -> str:
    torch_device = torch.device(device)
    model.eval()
    
    with torch.no_grad():
        output_chars = list(prefix.lower())
        hidden = None
        
        # Warm up hidden state with provided prefix
        for char in prefix.lower():
            if char not in char2idx:
                return prefix
            token = torch.tensor([[char2idx[char]]], dtype=torch.long, device=torch_device)
            logits, hidden = model(token, hidden)

        last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        next_char = idx2char[last_token]

        # Greedily generate subsequent characters until end token '$'
        while next_char != "$" and len(output_chars) < max_length:
            output_chars.append(next_char)
            token = torch.tensor([[last_token]], dtype=torch.long, device=torch_device)
            logits, hidden = model(token, hidden)
            last_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            next_char = idx2char[last_token]

        return "".join(output_chars)
