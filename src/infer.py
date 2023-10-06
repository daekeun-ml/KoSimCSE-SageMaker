import torch

def show_embedding_score(tokenizer, model, sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs)
    score01 = cal_score(embeddings[0,:], embeddings[1,:])
    score02 = cal_score(embeddings[0,:], embeddings[2,:])

    print(score01, score02)

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100