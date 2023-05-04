import torch
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

#-----------------------------------------------------------------------------------------------------------------
# load corpora

print("Loading corpora...")

dict_folder = "corpora/"
with open(dict_folder + 'duration_corpus.pkl', 'rb') as f:
    duration_corpus = pickle.load(f)
with open(dict_folder + 'offset_corpus.pkl', 'rb') as f:
    offset_corpus = pickle.load(f)

with open(dict_folder + "token_songs.pkl", "rb") as f:
    token_songs = pickle.load(f)
max_num_tokens = len(duration_corpus) * len(offset_corpus)

#-----------------------------------------------------------------------------------------------------------------
# create corpus of duration/offset combinations

timing_corpus = dict()

for dur in duration_corpus.values():
    for off in offset_corpus.values():
        timing_corpus[".".join([str(dur),str(off)])] = len(timing_corpus)
        
#-----------------------------------------------------------------------------------------------------------------
# create count matrix

print("Creating count matrix")

count_matrix = torch.zeros(size=(max_num_tokens, max_num_tokens), device=device, dtype=torch.float32)

window_size = 4

for song in tqdm(token_songs):
    song_length = len(song)
    for idx, token in enumerate(song):
        curr_token = ".".join(token.split('.')[1:3])
        curr_token_id = timing_corpus[curr_token]

        prior = max(0,idx-window_size)
        for s in range(prior, idx):
            distance = idx-prior
            context_token = ".".join(song[s].split('.')[1:3])
            context_token_id = timing_corpus[context_token]

            count_matrix[curr_token_id, context_token_id] += 1/distance
            count_matrix[context_token_id, curr_token_id] += 1/distance

torch.save(count_matrix, f"timing_matrix_{window_size}.pt")

#-----------------------------------------------------------------------------------------------------------------
# show frequencies

sums = (1+count_matrix.sum(dim=1)).log()
plt.figure(0, figsize=(20,10))
plt.plot(sums.cpu(), 'bo')
plt.title("Timing Frequencies")
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# Cut count matrix to important pairings

info_indices = [i for i,s in enumerate(sums) if s != 0]
count_matrix = count_matrix[info_indices][:,info_indices]

keys = list(timing_corpus.keys())
new_timing_corpus = dict()
for i in info_indices:
    new_timing_corpus[keys[i]] = len(new_timing_corpus)
timing_corpus = new_timing_corpus

torch.save(count_matrix, f"timing_matrix_{window_size}.pt")
with open("./corpora/timing_corpus.pkl", "wb") as f:
    pickle.dump(timing_corpus, f)

num_tokens = len(info_indices)

#-----------------------------------------------------------------------------------------------------------------
# Prepare parameters

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

embedding_dim = 53

vectors = 0.2*(2*torch.rand(size=(2, num_tokens, embedding_dim), device=device)-1)
biases = torch.zeros(size=(2, num_tokens), device=device)

vectors.requires_grad = True
biases.requires_grad = True
params = [vectors, biases]
optimizer = torch.optim.Adagrad(params=params, lr=2e-1)

#-----------------------------------------------------------------------------------------------------------------
# train with GloVe algorithm

losses = []

num_epochs = 100
batch_size = 1_000
num_batches = num_nonzero // batch_size


for epoch in tqdm(range(num_epochs)):

    if biases.sum() == torch.nan:
        print("NAN")
        break
    
    epoch_loss = 0
    r = torch.randperm(num_nonzero)
    for batch in range(num_batches):

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

#-----------------------------------------------------------------------------------------------------------------
# show losses

print(losses)
plt.semilogy(losses)
plt.title("Epoch losses")
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# saving params

print("Saving parameters...")

torch.save(vectors, f"embeddings/timing/dim53/vector100.pt")
torch.save(biases, f"embeddings/timing/dim53/biases100.pt")
torch.save(optimizer, f"embeddings/timing/dim53/optimizer100.pt")
torch.save(optimizer.state_dict(), f"embeddings/timing/dim53/optimizer_state100.pt")
with open(f"embeddings/timing/dim53/losses100.pkl", "wb") as f:
    pickle.dump(losses, f)
