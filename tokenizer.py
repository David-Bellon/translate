import pandas as pd

df = pd.read_csv("translate.csv")
words_en = set()
words_es = set()
for line in df["En"]:
    for word in line.split(" "):
        words_en.add(word)
for line in df["Es"]:
    for word in line.split(" "):
        words_es.add(word)

print(words_en)
with open("en_vocab.txt", "a") as f:
    for i, w in enumerate(list(words_en)[1:]):
        f.write(w + " " + str(i) + "\n")

with open("es_vocab.txt", "a") as f:
    for i, w in enumerate(list(words_es)[1:]):
        f.write(w + " " + str(i) + "\n")