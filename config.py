import torch

# The local directory path where downloaded data will be saved.
DOWNLOAD_PATH = "data"

# Reranker 
RERANKER = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'

# The identifier of the pre-trained sentence transformer model for producing sentence embeddings.
MODEL_SENTENCE_TRANSFORMER = 'paraphrase-MiniLM-L3-v2'

# The computing device to be used for model inference and training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

