import string
import pandas as pd
df = pd.DataFrame(columns=["En", "Es"])
with open("spa-eng/spa.txt") as f:
    en = []
    es = []
    for line in f:
        x = line.translate(str.maketrans('', '', string.punctuation)).replace("¿", "").replace("¡", "").lower()
        _en, _es = x.split("\t")[:2]
        en.append(_en)
        es.append(_es)
df["En"] = en
df["Es"] = es
df.to_csv("translate.csv", index=False)