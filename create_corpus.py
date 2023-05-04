import os
from music21 import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from fractions import Fraction
from operator import itemgetter
import pickle
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------------------------
# Load song data
def get_songs(folder_path: str) -> list:
    songs = []
    failed_songs = []
    for artist in tqdm(os.listdir(folder_path)):
        artist_path = folder_path + artist + "/"
        for song_name in os.listdir(artist_path):
            try:
                song = converter.parse(artist_path + song_name)
                songs.append(song.flatten())
            except:
                failed_songs.append(artist_path + song_name)

    return songs, failed_songs
  
folder_path = "./midis/"
print("Loading songs...")
loaded_songs, failed_songs = get_songs(folder_path)
print(f"Loaded ({len(loaded_songs)}) songs")
print(f"Failed to load ({len(failed_songs)}) song files: {failed_songs}")

songs = [song.notes for song in loaded_songs]

#-----------------------------------------------------------------------------------------------
# Collect data on durations, offsets, pitches, and volumes

# To collect information on the types of durations and offsets, and their frequencies
durations_dict = dict()
offsets_dict = dict()
pitches_dict = dict()
volumes_dict = dict()


max_volume_denom = 12
max_duration_denom = 12
max_offset_denom = 12

print("Collecting info on the types of durations and offsets, and their frequencies...")
for song in tqdm(songs):
    prev_offset = 0
    for item in song:

        # add volume info
        v = item.volume.velocityScalar
        v = Fraction(int(round(max_volume_denom*v)),max_volume_denom)
        volumes_dict[v] = volumes_dict.get(v,0) + 1

        # add pitch info
        if isinstance(item, note.Note):
            p = str(item.pitch)
            pitches_dict[p] = pitches_dict.get(p, 0) + 1
        elif isinstance(item, chord.Chord):
            p = [str(y) for y in item.pitches]
            p.sort()
            p = " ".join(p)
            pitches_dict[p] = pitches_dict.get(p, 0) + 1

        # add duration info
        d = item.duration.quarterLength
        d = Fraction(int(round(max_duration_denom*r)),max_duration_denom)
        durations_dict[d] = durations_dict.get(d, 0) + 1

        # add relative offset info
        r = item.offset-prev_offset
        r = Fraction(int(round(max_offset_denom*r)),max_offset_denom)
        offsets_dict[r] = offsets_dict.get(r, 0) + 1
        prev_offset = item.offset

durations_dict = dict(sorted(durations_dict.items()))
offsets_dict = dict(sorted(offsets_dict.items()))
pitches_dict = dict(sorted(pitches_dict.items()))
volumes_dict = dict(sorted(volumes_dict.items()))

#-----------------------------------------------------------------------------------------------
# Show frequencies of durations, offsets, and volumes

fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))
fig.suptitle('Histograms of Properties')
ax1.semilogy(volumes_dict.keys(), volumes_dict.values(), 'bo')
ax2.semilogy(offsets_dict.keys(), offsets_dict.values(), 'bo')
ax3.semilogy(durations_dict.keys(), durations_dict.values(), 'bo')

ax1.set_title("Volumes")
ax2.set_title("Relative Offsets")
ax3.set_title("Durations")

ax4.bar([k for k in volumes_dict.keys()], [np.log(v) for v in volumes_dict.values()], width=[.15])
ax5.bar([k for k in offsets_dict.keys()], [np.log(v) for v in offsets_dict.values()], width=[.25])
ax6.bar([k for k in durations_dict.keys()], [np.log(v) for v in durations_dict.values()], width=[.25])

plt.show()

#-----------------------------------------------------------------------------------------------
# show inclusion rates for different frequency cutoffs of pitches

cutoffs = np.arange(1,500)

total_amt = np.sum([v for v in pitches_dict.values()])
total_unique = len(pitches_dict)

ignored_list = []
unique_list = []
for cutoff in cutoffs:
    ignored_amt = 0
    unique_amt = 0
    for val in pitches_dict.values():
        if val < cutoff:
            ignored_amt += val
        else:
            unique_amt += 1
    ignored_list.append(ignored_amt/total_amt)
    unique_list.append(unique_amt/total_unique)

plt.semilogx(cutoffs, ignored_list)
plt.semilogx(cutoffs, unique_list)
plt.legend(["Percent Ignored", "Percent Unique Retained"])
plt.title("Inclusion of Pitches Tokens for Given Frequency Cutoffs")
plt.show()

#-----------------------------------------------------------------------------------------------
# Create corpora

duration_corpus = dict()
offset_corpus = dict()
volume_corpus = dict()
pitches_corpus = {"<NULL>": 0}

duration_cutoff = 2
offset_cutoff = 2
volume_cutoff = 2
pitches_cutoff = 2

for key,val in durations_dict.items():
    if not val < duration_cutoff:
        duration_corpus[key] = len(duration_corpus)

for key,val in offsets_dict.items():
    if not val < offset_cutoff:
        offset_corpus[key] = len(offset_corpus)

for key,val in volumes_dict.items():
    if not val < volume_cutoff:
        volume_corpus[key] = len(volume_corpus)

for key,val in pitches_dict.items():
    if not val < pitches_cutoff:
        pitches_corpus[key] = len(pitches_corpus)


print(f"There are {len(duration_corpus)} tokens in the duration corpus, which is {100*len(duration_corpus)/len(durations_dict):.2f}% of the total duration tokens, but accounts for {100*np.sum(itemgetter(*list(duration_corpus.keys()))(durations_dict))/np.sum(list(durations_dict.values())):.2f}% of corpus")
print(f"There are {len(offset_corpus)} tokens in the offset corpus, which is {100*len(offset_corpus)/len(offsets_dict):.2f}% of the total offset tokens, but accounts for {100*np.sum(itemgetter(*list(offset_corpus.keys()))(offsets_dict))/np.sum(list(offsets_dict.values())):.2f}% of corpus")
print(f"There are {len(volume_corpus)} tokens in the volume corpus, which is {100*len(volume_corpus)/len(volumes_dict):.2f}% of the total volume tokens, but accounts for {100*np.sum(itemgetter(*list(volume_corpus.keys()))(volumes_dict))/np.sum(list(volumes_dict.values())):.2f}% of corpus")
print(f"There are {len(pitches_corpus)} tokens in the pitches corpus, which is {100*len(pitches_corpus)/len(pitches_dict):.2f}% of the total pitches tokens, but accounts for {100*np.sum(itemgetter(*list(pitches_corpus.keys())[1:])(pitches_dict))/np.sum(list(pitches_dict.values())):.2f}% of corpus")


#-----------------------------------------------------------------------------------------------
# Convert midi songs to songs of tokens

full_corpus = dict()

token_songs = []

print("Converting midi songs to tokens...")

for song in tqdm(songs):
    token_song = []
    prev_offset = 0
    for item in song:

        # add volume info
        v = item.volume.velocityScalar

        # add pitch info
        if isinstance(item, note.Note):
            p = str(item.pitch)
        elif isinstance(item, chord.Chord):
            p = [str(y) for y in item.pitches]
            p.sort()
            p = " ".join(p)

        # add duration info
        d = item.duration.quarterLength

        # add relative offset info
        r = item.offset-prev_offset
        prev_offset = item.offset

        # convert property values to token ids
        if not p in pitches_corpus:
            p = "<NULL>"
        if not d in duration_corpus:
            d = min(duration_corpus.keys(), key = lambda x: abs(x-d))
        if not r in offset_corpus:
            r = min(offset_corpus.keys(), key = lambda x: abs(x-r))
        if not v in volume_corpus:
            v = min(volume_corpus.keys(), key = lambda x: abs(x-v))
        
        p_id = str(pitches_corpus[p])
        d_id = str(duration_corpus[d])
        r_id = str(offset_corpus[r])
        v_id = str(volume_corpus[v])

        # add item token to full corpus
        full_token = ".".join([p_id, d_id, r_id, v_id])
        if not full_token in full_corpus:
            full_corpus[full_token] = len(full_corpus)
        
        # add token to token song
        token_song.append(full_token)
    # add token song to token songs
    token_songs.append(token_song)

#-----------------------------------------------------------------------------------------------
# Save corpora and songs

print("Saving corpora...")

dict_folder = "./corpora/"
if not os.path.exists(dict_folder):
    os.makedirs(dict_folder)

f = open(dict_folder + "full_corpus.pkl", "wb")
pickle.dump(full_corpus, f)
f.close()

f = open(dict_folder + "duration_corpus.pkl", "wb")
pickle.dump(duration_corpus, f)
f.close()

f = open(dict_folder + "offset_corpus.pkl", "wb")
pickle.dump(offset_corpus, f)
f.close()

f = open(dict_folder + "volume_corpus.pkl", "wb")
pickle.dump(volume_corpus, f)
f.close()

f = open(dict_folder + "pitches_corpus.pkl", "wb")
pickle.dump(pitches_corpus, f)
f.close()

f = open(dict_folder + "token_songs.pkl", "wb")
pickle.dump(token_songs, f)
f.close()

