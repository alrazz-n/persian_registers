import json
import time
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import expit as sigmoid
import json
import zstandard as zstd
import io

# ------------------------------------------------
# Labels
# ------------------------------------------------

labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

all_valid_labels = sorted(
    list(labels_structure.keys())
    + [s for subs in labels_structure.values() for s in subs]
)

#----------------------------
#Load JSONL
#-----------------------------


def load_jsonl_texts(filepath, limit=None):
    texts = []
    
    # 1. Open the file in binary mode ('rb')
    with open(filepath, 'rb') as f:
        # 2. Create a decompression context
        dctx = zstd.ZstdDecompressor()
        
        # 3. Create a stream reader to decompress on the fly
        with dctx.stream_reader(f) as reader:
            # 4. Wrap the binary stream in a TextIOWrapper so we can read lines
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            for i, line in enumerate(text_stream):
                if limit and i >= limit:
                    break
                
                try:
                    # Parse the JSON line
                    data = json.loads(line)
                    # HPLT data usually uses the key 'text' or 'content'
                    texts.append(data.get("text", "")) 
                except json.JSONDecodeError:
                    continue
                    
    return texts

# Usage
texts = load_jsonl_texts("/scratch/project_2005092/nima/10_1.jsonl.zst", limit=1_000)
print("Samples loaded:", len(texts))



#def count_jsonl_lines(filepath):
#    count = 0
#    with open(filepath, 'rb') as f:
#        dctx = zstd.ZstdDecompressor()
#        with dctx.stream_reader(f) as reader:
#            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
#            for _ in text_stream:
#                count += 1
#    return count

#total_samples = count_jsonl_lines("./10_1.jsonl.zst")
#print(f"Total samples in file: {total_samples:,}")
#First shard was 104,762#

#Make dataset
dataset = Dataset.from_dict({"text": texts})

#-------------------
#Tokenization
#---------------------


model_name = "FacebookAI/xlm-roberta-large" 
print (f"model name:{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # If it still crashes, try 256
    )

dataset = dataset.map(
    tokenize, 
    batched=True, 
    batch_size=100, 
    remove_columns=["text"] 
)

dataset.set_format("torch", columns=["input_ids", "attention_mask"])

print("Success! Dataset is ready.")

#-----------------
#load model
#--------------
model = AutoModelForSequenceClassification.from_pretrained(
    "./test_train/fine_tuned_xlmr_persian",  
    num_labels=len(all_valid_labels),
    problem_type="multi_label_classification",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#--------------
#inference
#-------------
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)
# Warm-up (important on GPU)
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
        break

# Timing
start = time.time()

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs.logits)

end = time.time()

elapsed = end - start
samples_per_sec = len(dataset) / elapsed

print(f"Elapsed time: {elapsed:.2f} sec")
print(f"Throughput: {samples_per_sec:.1f} samples/sec")

#---------------------------
#estimate
#------------------------
total_samples = 104_762  # example
estimated_time = total_samples / samples_per_sec

print(f"Estimated labeling time: {estimated_time/3600:.2f} hours")
