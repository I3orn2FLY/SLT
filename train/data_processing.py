import spacy
import pandas as pd
import pickle
import glob
import numpy as np
import re
import os
from numpy.random import shuffle, seed

seed(0)


class Vocab:
    def __init__(self, sentences):
        self.wtoi = {"<sos>": 0, "<eos>": 1, "<unk>": 2, "<pad>": 3}
        self.itow = None
        self.word_count = {}
        self.sentences = sentences
        self.sent_lengths = None

    def build(self, min_freq=2):
        for sentence in self.sentences:
            for tok in sentence:
                if not self.word_count.get(tok):
                    self.word_count[tok] = 1
                else:
                    self.word_count[tok] += 1

        for word in list(self.word_count):
            if self.word_count[word] < min_freq:
                del self.word_count[word]

        idx = 4
        for word in self.word_count:
            self.wtoi[word] = idx
            idx += 1

        self.itow = {i: w for w, i in self.wtoi.items()}

        self.sent_lengths = pd.Series([len(sentence) for sentence in self.sentences])

    def sentence_to_num(self, sentence, max_length):
        if len(sentence) > max_length - 2:
            return None
        idxs = [self.wtoi["<sos>"]]
        for tok in sentence:
            idx = self.wtoi.get(tok)
            if idx is None: idx = self.wtoi["<unk>"]
            idxs.append(idx)
        idxs.append(self.wtoi["<eos>"])
        for i in range(len(sentence) + 2, max_length):
            idxs.append(self.wtoi["<pad>"])

        return idxs

    def plot_sent_lengths(self):
        print(self.sent_lengths.value_counts())
        pass

    def save_vocab(self, filename="../vars/vocab.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def load_sentences(filename="../data/manual/PHOENIX-2014-T.train.corpus.csv", save=False):
    df = pd.read_csv(filename, sep='|')
    spacy_de = spacy.load("de")
    sentences = []
    for sentence in df.translation:
        tokens = [tok.lower_ for tok in spacy_de.tokenizer(sentence)]
        sentences.append(tokens)
    if save:
        with open("../vars/sents.pkl", "wb") as f:
            pickle.dump(sentences, f)
    return sentences


class PheonixDataset():
    def __init__(self, vocab):
        self.targets = []
        self.features = []
        self.vocab = vocab

    def generate_dataset(self, datapath="/home/kenny/Workspace/Data/SLT", side=227, split="train"):
        filename = datapath + "/manual/PHOENIX-2014-T." + split + ".corpus.csv"

        df = pd.read_csv(filename, sep="|")
        prefix = datapath + "/open_pose/fullFrame-" + str(side) + "x" + str(side) + "px/" + split + "/"
        spacy_de = spacy.load("de")

        for idx in range(df.shape[0]):
            row = df.iloc[idx]

            tokens = [tok.lower_ for tok in spacy_de.tokenizer(row.translation)]
            nums = self.vocab.sentence_to_num(tokens, 50)
            if nums is None: continue
            self.targets.append(nums)

            path = prefix + row.video
            path = path.replace("1/*.png", "*.pkl")
            video_pose = []
            frame_nums = []
            for pose_filename in glob.glob(path):
                with open(pose_filename, 'rb') as f:
                    pose_data = pickle.load(f)

                    face = pose_data['face'].reshape(-1)
                    body = pose_data['body'].reshape(-1)
                    lhand = pose_data['left_hand'].reshape(-1)
                    rhand = pose_data['right_hand'].reshape(-1)
                    pose = np.hstack((face, body, lhand, rhand))

                    pose_filename = os.path.split(pose_filename)[-1]
                    frame_n = int(re.findall(r'([0-9]+)', pose_filename)[-1])

                    video_pose.append(pose)
                    frame_nums.append(frame_n)

            video_pose = np.array(video_pose)
            video_pose = video_pose[np.argsort(frame_nums)]

            self.features.append(video_pose)

            if idx % 50 == 0:
                percent = idx / df.shape[0] * 100
                print("\rProgress ", percent, end=" ")

        print()
        print(len(self.features))
        print(len(self.targets))

    def process_features(self):
        n_frames = pd.Series([len(feature) for feature in self.features])
        val_counts = n_frames.value_counts()
        pass

    def save_base(self, filename="../vars/base.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    # with open("../vars/vocab.pkl", "rb") as f:
    #     vocab = pickle.load(f)

    with open("../vars/base.pkl", "rb") as f:
        dataset = pickle.load(f)
     # = PheonixDataset(vocab)

    dataset.process_features()
    # dataset.generate_dataset()

    # dataset.save_base()

    # sentences = load_sentences(save=True)
    # vocab = Vocab(sentences)
    #
    # vocab.build()
    # vocab.save_vocab()
    # print(len(vocab.word_count))
    # pass
