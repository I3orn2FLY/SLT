import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
import numpy as np
import pickle
from numpy.random import shuffle
from data_processing import Vocab
from nltk.translate.bleu_score import corpus_bleu
import warnings


# TODO
#  1. Try Pheonix Bleu Score Done
#  2. Try different sequence length
#  3. Change fill sample
#  4. New dataset

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def get_tensor_batch(X, y, batch_idx, batch_size, device):
    start = batch_idx * batch_size
    end = start + batch_size

    if end > X.shape[1]: end = X.shape[1]
    X_batch = torch.Tensor(X[:, start:end, :]).to(device)
    y_batch = torch.LongTensor(y[:, start:end]).to(device)

    return X_batch, y_batch


def get_bleu_score(trg, output, vocab):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx_sentences_hyp = torch.t(output.argmax(dim=2)).cpu().numpy()
        idx_sentences_ref = torch.t(trg).cpu().numpy()

        hypotheses = vocab.num_to_sentence_batch(idx_sentences_hyp)
        references = vocab.num_to_sentence_batch(idx_sentences_ref)
        references = [[sentence] for sentence in references]

        return corpus_bleu(references, hypotheses)


def eval(model, src, trg, criterion, vocab, beam_size):
    model.eval()

    with torch.no_grad():
        src = torch.Tensor(src).to(model.device)
        trg = torch.LongTensor(trg).to(model.device)

        output = model(src, trg, 0, beam_size)

        bleu = get_bleu_score(trg, output, vocab)

        output = output[1:].view(-1, output.shape[-1])

        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

        return loss.item(), bleu


def train(model, X_train, Y_train, X_val, y_val, X_test, y_test, vocab, learning_rate, criterion, batch_size, n_epochs,
          clip,
          load=True, save=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    device = model.device
    num_batches = X_train.shape[1] // batch_size
    if X_train.shape[1] % batch_size != 0: num_batches += 1

    train_loss = 0
    train_bleu = 0
    best_bleu = 0

    if load:
        model.load_state_dict(torch.load("../weights/convfeat2trans/weights.pt"))
        with open("../weights/convfeat2trans/best_bleu.txt", 'r') as f:
            best_bleu = float(f.readline())
            print("Best bleu:", best_bleu)

    for epoch in range(1, n_epochs + 1):
        model.train()

        for batch_idx in range(num_batches):
            src, trg = get_tensor_batch(X_train, Y_train, batch_idx, batch_size, device)

            optimizer.zero_grad()

            output = model(src, trg)
            bleu = get_bleu_score(trg, output, vocab)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            train_bleu += bleu
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            train_loss += loss.item()

            print("\rBatch Progress:" + str(batch_idx) + "/" + str(num_batches),
                  "Loss:", loss.item(),
                  "Bleu:", bleu, end=" ")

        train_loss = train_loss / num_batches
        train_bleu = train_bleu / num_batches

        val_loss, val_bleu = eval(model, X_val, y_val, criterion, vocab, beam_size=3)

        test_loss, test_bleu = eval(model, X_test, y_test, criterion, vocab, beam_size=3)

        print("\rEpoch", epoch,
              "Train Bleu: ", train_bleu,
              "Val Bleu:", val_bleu,
              "Test Bleu:", test_bleu)

        if val_bleu > best_bleu:

            best_bleu = val_bleu
            if save:
                torch.save(model.state_dict(), "../weights/convfeat2trans/weights.pt")
                with open("../weights/convfeat2trans/best_bleu.txt", 'w') as f:
                    f.write(str(best_bleu))
                print("Saved weights")


if __name__ == "__main__":
    data_prefix = "../vars/data/conv_feats/100to30/"
    X_train = np.load(data_prefix + "AugFactor_1/X_train.npy").transpose([1, 0, 2])
    y_train = np.load(data_prefix + "AugFactor_1/y_train.npy").transpose()

    X_val = np.load(data_prefix + "X_val.npy").transpose([1, 0, 2])
    y_val = np.load(data_prefix + "y_val.npy").transpose()

    X_test = np.load(data_prefix + "X_test.npy").transpose([1, 0, 2])
    y_test = np.load(data_prefix + "y_test.npy").transpose()

    with open("../vars/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT_DIM = 2048
    OUTPUT_DIM = len(vocab.itow)
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = ConvFeats2Trans(enc, dec, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.wtoi["<pad>"])

    model.apply(init_weights)

    train(model, X_train, y_train, X_val, y_val, X_test, y_test,
          vocab=vocab,
          learning_rate=0.001,
          criterion=criterion,
          batch_size=8,
          n_epochs=100,
          clip=1,
          load=True,
          save=True)
