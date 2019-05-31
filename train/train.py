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


# TODO: Implement Beam Search and apply Bleu-Score for evaluation


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


def eval(model, src, trg, criterion):
    model.eval()

    with torch.no_grad():
        src = torch.Tensor(src).to(model.device)
        trg = torch.LongTensor(trg).to(model.device)
        output = model(src, trg, 0)
        output = output[1:].view(-1, output.shape[-1])

        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

    return loss.item()


def train(model, X_train, Y_train, X_val, y_val, learning_rate, criterion, batch_size, n_epochs, clip):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = model.device
    num_batches = X_train.shape[1] // batch_size
    if X_train.shape[1] % batch_size != 0: num_batches += 1

    train_loss = 0

    best_loss = 10000
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_idx in range(num_batches):
            src, trg = get_tensor_batch(X_train, Y_train, batch_idx, batch_size, device)

            optimizer.zero_grad()

            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            train_loss += loss.item()

            print("\rBatch Progress:" + str(batch_idx) + "/" + str(num_batches), "Loss:", loss.item(), end=" ")

        train_loss = train_loss / num_batches

        val_loss = eval(model, X_val, y_val, criterion)

        print("\rEpoch", epoch, "Train Loss: ", train_loss, "Val Loss: ", val_loss)

        if val_loss < best_loss:
            print("Saved weights")
            best_loss = val_loss
            torch.save(model.state_dict(), '../weights/weights.pt')


if __name__ == "__main__":
    X_train = np.load("../vars/X_train.npy").transpose([1, 0, 2])
    y_train = np.load("../vars/y_train.npy").transpose()

    X_val = np.load("../vars/X_dev.npy").transpose([1, 0, 2])
    y_val = np.load("../vars/y_dev.npy").transpose()

    with open("../vars/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT_DIM = 110
    OUTPUT_DIM = len(vocab.itow)
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.wtoi["<pad>"])

    model.apply(init_weights)

    # model.load_state_dict(torch.load('../weights/weights.pt'))
    train(model, X_train, y_train, X_val, y_val, learning_rate=0.001, criterion=criterion, batch_size=256, n_epochs=100,
          clip=10)
