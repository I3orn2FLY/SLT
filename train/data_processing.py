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

    def generate_dataset(self, datapath="/home/kenny/Workspace/Data/SLT",
                         side=227,
                         with_face=False,
                         split="train",
                         input_seq_length=50,
                         augment_factor=1,
                         out_seq_length=50):
        filename = datapath + "/manual/PHOENIX-2014-T." + split + ".corpus.csv"

        df = pd.read_csv(filename, sep="|")
        prefix = datapath + "/open_pose/fullFrame-" + str(side) + "x" + str(side) + "px/" + split + "/"
        spacy_de = spacy.load("de")

        for idx in range(df.shape[0]):
            row = df.iloc[idx]

            tokens = [tok.lower_ for tok in spacy_de.tokenizer(row.translation)]
            nums = self.vocab.sentence_to_num(tokens, out_seq_length)
            if nums is None: continue

            path = prefix + row.video
            path = path.replace("1/*.png", "*.pkl")
            video_pose = []
            frame_nums = []
            for pose_filename in glob.glob(path):
                with open(pose_filename, 'rb') as f:
                    pose_data = pickle.load(f)

                    body_idxs = list(range(9)) + list(range(15, 19))
                    body = pose_data['body']

                    if body.shape[0] > 1 or body.shape[0] < 1:
                        continue

                    body = body.squeeze()[body_idxs]
                    lhand = pose_data['left_hand'].squeeze()
                    rhand = pose_data['right_hand'].squeeze()

                    pose = np.vstack((body, lhand, rhand))
                    if with_face:
                        face = pose_data['face'].squeeze()
                        pose = np.vstack((face, pose))
                    pose = pose[:, :2] / side

                    pose = pose.reshape(-1)

                    pose_filename = os.path.split(pose_filename)[-1]
                    frame_n = int(re.findall(r'([0-9]+)', pose_filename)[-1])

                    video_pose.append(pose)
                    frame_nums.append(frame_n)

            video_pose = np.array(video_pose)

            aug_samples = PheonixDataset.augment_sample(video_pose, augment_factor, input_seq_length)

            self.targets += len(aug_samples) * [nums]
            self.features += aug_samples

            if idx % 10 == 0:
                percent = idx / df.shape[0] * 100
                print("\rProgress %.2f" % percent, end=" ")

        print()
        print(len(self.targets))

    @staticmethod
    def augment_sample(sample, aug_factor, seq_length):
        aug = []
        L = len(sample)

        if (L < seq_length):
            last_frame = sample[-1]
            fill_frames = []
            needed = seq_length - len(sample)
            for i in range(needed):
                fill_frames.append(last_frame.copy())

            fill_frames = np.array(fill_frames)
            aug.append(np.vstack((sample, fill_frames)))

        else:
            step = L // seq_length
            left_over = L % seq_length
            starting_idxs = list(range(step + left_over))
            shuffle(starting_idxs)
            for idx, start in enumerate(starting_idxs):
                seq = [start + i * step for i in range(seq_length)]

                aug.append(sample[seq])
                if aug_factor <= idx + 1: break

        return aug

    def save_XY(self, path, split, augment_factor):
        X = np.array(self.features)
        y = np.array(self.targets)
        idxs = list(range(y.shape[0]))
        shuffle(idxs)

        X = X[idxs]
        y = y[idxs]
        y = y.astype(np.int32)
        np.save(os.path.join(path, "X_" + split + str(augment_factor)), X)
        np.save(os.path.join(path, "y_" + split + str(augment_factor)), y)


if __name__ == "__main__":
    with open("../vars/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    split = "train"
    aug_factor = 1
    dataset = PheonixDataset(vocab)
    dataset.generate_dataset(split=split, augment_factor=aug_factor)
    dataset.save_XY("../vars/", split=split, augment_factor=aug_factor)

    # sentences = load_sentences(save=True)
    # vocab = Vocab(sentences)
    #
    # vocab.build()
    # vocab.save_vocab()
    # print(len(vocab.word_count))
    # pass
