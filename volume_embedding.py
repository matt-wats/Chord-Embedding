import torch
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

#---------------------------------------------------------------------------------------
# Load corpus

dict_folder = "corpora/"
with open(dict_folder + 'volume_corpus.pkl', 'rb') as f:
    volume_corpus = pickle.load(f)

with open(dict_folder + "token_songs.pkl", "rb") as f:
    token_songs = pickle.load(f)

num_tokens = len(volume_corpus)

#---------------------------------------------------------------------------------------
# create count matrix

print("Creating count matrix...")

count_matrix = torch.zeros(size=(num_tokens, num_tokens), device=device, dtype=torch.float32)

window_size = 4

for song in tqdm(token_songs):
    song_length = len(song)
    for idx, token in enumerate(song):
        curr_token_id = int(token.split('.')[3])

        prior = max(0,idx-window_size)
        for s in range(prior, idx):
            distance = idx-prior
            context_token_id = int(song[s].split('.')[3])

            count_matrix[curr_token_id, context_token_id] += 1/distance
            count_matrix[context_token_id, curr_token_id] += 1/distance

torch.save(count_matrix, f"volume_matrix_{window_size}.pt")

#---------------------------------------------------------------------------------------
# show frequencies

sums = (1+count_matrix.sum(dim=1)).log()
plt.figure(0, figsize=(20,10))
plt.plot(sums.cpu(), 'bo')
plt.title("Volume Frequencies")
plt.show()

#---------------------------------------------------------------------------------------
# prepare params

def weight_func(x):
    x_max = 100
    alpha = 3/4

    if x > x_max:
        return 1.0
    else:
        return (x/x_max)**alpha

nonzeros = count_matrix.nonzero()
num_nonzero = nonzeros.size(0)
print(f"There are {num_nonzero} entries in the count matrix")

embedding_dim = 3

vectors = 0.2*(2*torch.rand(size=(2, num_tokens, embedding_dim), device=device)-1)
biases = torch.zeros(size=(2, num_tokens), device=device)

vectors.requires_grad = True
biases.requires_grad = True
params = [vectors, biases]
optimizer = torch.optim.Adagrad(params=params, lr=2e-1)

#---------------------------------------------------------------------------------------
# train using GloVe algorithm

losses = []

num_epochs = 100
batch_size = 10
num_batches = num_nonzero // batch_size

print("Starting training...")

for epoch in range(num_epochs):

    if biases.sum() == torch.nan:
        print("NAN")
        break
    
    epoch_loss = 0
    r = torch.randperm(num_nonzero)
    for batch in (range(num_batches)):

        batch_elements = nonzeros[r[batch*batch_size:(batch+1)*batch_size]]

        optimizer.zero_grad()

        for i,j in batch_elements:
            x = count_matrix[i,j]

            w = vectors[0, i]
            w_hat = vectors[1,j]
            b = biases[0,i]
            b_hat = biases[1,j]

            loss_term = (w.dot(w_hat) + b + b_hat - x.log())
            weight_term = weight_func(x)
            batch_loss = weight_term * (loss_term**2)
            batch_loss.backward()
            epoch_loss += batch_loss.item()

        optimizer.step()

    losses.append(epoch_loss)

    print(f"Just completed epoch #{epoch+1} / {num_epochs}, with an epoch loss: {epoch_loss:.2f}")

#---------------------------------------------------------------------------------------
# show losses

print(losses) #prev best 2e-1, 64, ~300, 128iter
plt.semilogy(losses)
plt.title("Epoch losses")
plt.show()

#---------------------------------------------------------------------------------------
# save params

torch.save(vectors, f"./embeddings/volume/dim3/vector100.pt")
torch.save(biases, f"./embeddings/volume/dim3/biases100.pt")
torch.save(optimizer, f"./embeddings/volume/dim3/optimizer100.pt")
torch.save(optimizer.state_dict(), f"./embeddings/volume/dim3/optimizer_state100.pt")
with open(f"./embeddings/volume/dim3/losses100.pkl", "wb") as f:
    pickle.dump(losses, f)
