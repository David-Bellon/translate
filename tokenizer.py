from transformers import DistilBertTokenizer, AutoTokenizer
import pandas as pd

df = pd.read_csv("translate.csv")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
