'''
Author: Hannah
Date: 2026-01-20 21:09:20
LastEditTime: 2026-01-21 21:12:20
'''
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):

    def __init__(self, data):
        self.data = data
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
        center, context = self.data[idx]
        center_tensor = torch.tensor(center, dtype=torch.long)
        context_tensor = torch.tensor(context, dtype=torch.long)

        return center_tensor, context_tensor

# Simple Skip-gram Module
class Word2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)

        initrange = 0.5 / embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, center_words, context_words):
        emb_u = self.u_embeddings(center_words)
        emb_v = self.v_embeddings(context_words)
        
        score = torch.bmm(emb_u.unsqueeze(1), emb_v.transpose(1, 2)).squeeze(1)

        return score

    def get_embeddings(self):
        return self.u_embeddings.weight.detach().cpu()

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

sent_list = data['sent_list']
counter = data['counter']
word2idx = data['word2idx']
idx2word = data['idx2word']
skipgram_df = data['skipgram_df']
vocab_size = len(word2idx)


# Precompute negative sampling distribution below
neg_sample_dist = torch.tensor([counter[idx2word[i]] for i in range(len(idx2word))], dtype=torch.float)
neg_sample_dist = neg_sample_dist.pow(0.75)
neg_sample_dist = neg_sample_dist / neg_sample_dist.sum()

# Device selection: CUDA > MPS > CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset and DataLoader
datset = SkipGramDataset(skipgram_df.values)
dataloader = DataLoader(datset, batch_size=BATCH_SIZE, shuffle=True)


# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def make_targets(center, context):

    batch_size = center.size(0)
    num_samples = context.size(1) 
    
    targets = torch.zeros(batch_size, num_samples).to(center.device)
    targets[:, 0] = 1.0
    
    return targets

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        center = center.to(device)
        context = context.to(device)
        batch_size = center.size(0)

        # Negative sampling
        neg_samples = torch.multinomial(neg_sample_dist, batch_size * NEGATIVE_SAMPLES, replacement=True).to(device)
        neg_samples = neg_samples.view(batch_size, NEGATIVE_SAMPLES)

        # Exclude positive samples from negative samples
        mask = (neg_samples == context.unsqueeze(1))
        # Resample until no collisions
        while mask.any():
            n_hits = mask.sum().item()
            new_samples = torch.multinomial(neg_sample_dist, n_hits, replacement=True).to(device)
            neg_samples[mask] = new_samples
            mask = (neg_samples == context.unsqueeze(1))

        # concat positive and negative samples
        context_combined = torch.cat((context.unsqueeze(1), neg_samples), dim=1)

        # targets
        targets = make_targets(center, context_combined)

        # Forward pass
        optimizer.zero_grad()
        output = model(center, context_combined)
        loss = loss_func(output, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")


# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
