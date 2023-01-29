from transformers import AutoTokenizer
import numpy as np

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data = np.load("./data/Urban/words.npy", allow_pickle=True)

word_data = data[0]
def_data = data[1]

inputs = tokenizer(word_data, def_data)
tokenizer.decode(inputs["input_ids"])

