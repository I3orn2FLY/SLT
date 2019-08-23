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

    def num_to_sentence(self, nums):
        sentence = []

        for num in nums:
            if num == self.wtoi["<pad>"] or num == self.wtoi["<sos>"]: continue
            if num == self.wtoi["<eos>"]: break

            sentence.append(self.itow[num])

        return sentence

    def num_to_sentence_batch(self, nums_batch):
        return [self.num_to_sentence(nums) for nums in nums_batch]

    def plot_sent_lengths(self):
        print(self.sent_lengths.value_counts())
        pass

    def save_vocab(self, filename="../vars/vocab.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def get_sentences(filename="../data/manual/PHOENIX-2014-T.train.corpus.csv", save=False):
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


class FeatureDataset():
    def __init__(self, vocab, aug_factor, input_seq_length=100, output_seq_length=50):

        self.vocab = vocab

        self.augment_factor = aug_factor
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

        self.spacy_de = spacy.load("de")

    def generate_split(self, split, datapath, side, with_face):
        print("Generating", split, "split.")
        anno_filename = datapath + "/manual/PHOENIX-2014-T." + split + ".corpus.csv"

        df = pd.read_csv(anno_filename, sep="|")

        image_path_prefix = datapath + "/resnet_feats/" + str(side) + "x" + str(side) + "/" + split + "/"

        X = []
        y = []

        for idx in range(df.shape[0]):
            row = df.iloc[idx]

            tokens = [tok.lower_ for tok in self.spacy_de.tokenizer(row.translation)]
            nums = self.vocab.sentence_to_num(tokens, self.output_seq_length)
            if nums is None: continue

            path = image_path_prefix + row.video
            path = path.replace("/1/*.png", ".npy")



            video_feats = np.load(path)

            aug_samples = self.augment_sample(video_feats, split=split)


            X += aug_samples
            y += len(aug_samples) * [nums]

            if idx % 10 == 0:
                percent = idx / df.shape[0] * 100
                print("\rProgress %.2f" % percent, end=" ")

        print()

        return np.array(X), np.array(y, dtype=np.int32)

    def generate_dataset(self, datapath="/home/kenny/Workspace/Data/SLT",
                         side=224, with_face=False,
                         only_train=False, save=False):

        args = {"datapath": datapath, "side": side, "with_face": with_face}
        X_train, y_train = self.generate_split(split="train", **args)
        if not only_train:
            X_val, y_val = self.generate_split(split="dev", **args)
            X_test, y_test = self.generate_split(split="test", **args)

        if save:
            save_path = "../vars/data/conv_feats/" + str(self.input_seq_length) + "to" + str(self.output_seq_length)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if not only_train:
                np.save(save_path + "/X_test", X_test)
                np.save(save_path + "/X_val", X_val)

                np.save(save_path + "/y_test", y_test)
                np.save(save_path + "/y_val", y_val)

            save_path += "/AugFactor_" + str(self.augment_factor)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + "/X_train", X_train)
            np.save(save_path + "/y_train", y_train)

            print("X Train_shape:", X_train.shape)
            print("Y train shape:", y_train.shape)

    def augment_sample(self, sample, split):
        aug = []
        L = len(sample)
        # TODO augmentation of dev and test set

        if (L < self.input_seq_length):
            # TODO change filling sample
            needed = self.input_seq_length - len(sample)
            fill_frames = needed * [(-1) * np.ones(sample[0].shape)]
            fill_frames = np.array(fill_frames)
            aug.append(np.vstack((sample, fill_frames)))

        else:
            step = L // self.input_seq_length
            left_over = L % self.input_seq_length
            starting_idxs = list(range(step + left_over))
            shuffle(starting_idxs)

            if split != "train":
                starting_idxs = [0]

            for idx, start in enumerate(starting_idxs):
                seq = [start + i * step for i in range(self.input_seq_length)]

                aug.append(sample[seq])
                if self.augment_factor <= idx + 1: break

        return aug

    def save_XY(self, path, split, augment_factor, input_seq_length, out_seq_length):
        X = np.array(self.features)
        y = np.array(self.targets)
        idxs = list(range(y.shape[0]))
        shuffle(idxs)

        X = X[idxs]
        y = y[idxs]
        y = y.astype(np.int32)
        np.save(os.path.join(path, "X_" + split + str(augment_factor) + "_[" + str(input_seq_length) + "to"), X)
        np.save(os.path.join(path, "y_" + split + str(augment_factor)), y)


if __name__ == "__main__":
    with open("../vars/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    dataset = FeatureDataset(vocab, aug_factor=1, input_seq_length=100, output_seq_length=30)
    dataset.generate_dataset(save=True)

    # sentences = get_sentences(save=False)
    # vocab = Vocab(sentences)

    # vocab.build()
    # vocab.save_vocab()
    # print(len(vocab.word_count))
    # pass
