import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from arch import Model
from tqdm import tqdm

MAX_LENGHT = 6
BATCH_SIZE = 100
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 0.0001
W_DECAY = 0.001
EPOCHS = 10

class Tokenizer():
    def __init__(self, txt_file, max_lenght):
        self.vocab = {}
        self.max_len = max_lenght
        with open(txt_file, "r", encoding="utf8") as f:
            for line in f:
                word, id = line.split(" ")
                self.vocab[word] = int(id.replace("\n", ""))

    def get_max_vocab_size(self):
        return list(self.vocab.values())[-1]

    def encode_text(self, text, origin):
        text = text.translate(str.maketrans('', '', string.punctuation)).replace("¿", "").replace("¡", "").lower()
        if origin:
            out = [self.vocab["[CLS]"]]
        else:
            out = []
        for word in text.split(" "):
            try:
                out.append(self.vocab[word])
            except:
                out.append(self.vocab["[UKN]"])
            if len(out) == self.max_len - 1:
                break
        if len(out) < self.max_len - 1:
            out = out + [self.vocab["[PAD]"]] * (self.max_len - len(out) - 1)
        out.append(self.vocab["[SEP]"])
        return out
    
    def decode_text(self, data):
        out = ""
        for id in data:
            if list(self.vocab.keys())[id] == "[SEP]":
                break
            out = out + " " + list(self.vocab.keys())[id]
        return out

en_tokenizer = Tokenizer("en_vocab.txt", MAX_LENGHT)
es_tokenizer = Tokenizer("es_vocab.txt", MAX_LENGHT)
df = pd.read_csv("translate.csv")

class MyData(Dataset):
    def __init__(self, data):
        super().__init__()
        self.en_data = list(data["En"])
        self.es_data = list(data["Es"])

    def __len__(self):
        return len(self.en_data)
    
    def __getitem__(self, index):
        en_word = self.en_data[index]
        es_word = self.es_data[index]

        en_word = en_tokenizer.encode_text(en_word, True)
        es_word = es_tokenizer.encode_text(es_word, False)
        if len(es_word) > 20:
            print(es_word)

        return torch.tensor(en_word).to(DEVICE), torch.tensor(es_word).to(DEVICE)
    
data = MyData(df)
train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])

train_set = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_set = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = Model(en_tokenizer.get_max_vocab_size(), es_tokenizer.get_max_vocab_size(), 512, 1024, 8, 2, 0.1, 0.5).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=W_DECAY)
loss_f = nn.CrossEntropyLoss()

def train(input, real):
    optimizer.zero_grad()

    out = model(input)
    loss = loss_f(out.transpose(1, 2), real)

    loss.backward()
    optimizer.step()

    return loss

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for i, (en_text, es_text) in tqdm(enumerate(train_set), total=len(train_set)):
        epoch_loss += train(en_text, es_text)
    print(f"Epoch: {epoch} ------ Loss: {epoch_loss.item()/i}")

torch.save(model, "translator.pt")