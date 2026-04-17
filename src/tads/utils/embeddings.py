"""Embedding model utilities for BGE-M3 encoding and similarity matching."""

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_embedding_model(model_name: str = "BAAI/bge-m3", device: str = None):
    """Load a sentence embedding model and tokenizer.

    Automatically uses DataParallel when multiple GPUs are available.

    Returns:
        (tokenizer, model, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs — using DataParallel.")
        model = torch.nn.DataParallel(model)

    print(f"Loaded {model_name} on {device}")
    return tokenizer, model, device


def encode_texts(
    texts: list,
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
    max_length: int = 64,
) -> torch.Tensor:
    """Encode texts into L2-normalised CLS-token embeddings.

    Args:
        texts: List of strings to encode.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model (or DataParallel wrapper).
        device: Device string ("cuda" / "cpu").
        batch_size: Encoding batch size.
        max_length: Max tokenizer length.

    Returns:
        Tensor of shape (len(texts), hidden_dim).
    """
    all_embeddings = []
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=max_length,
            ).to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


def match_to_tags_topk(
    text_embeds: torch.Tensor,
    tag_embeds: torch.Tensor,
    tag_texts: list,
    top_k: int = 3,
) -> tuple:
    """Find the top-k most similar tags for each text embedding.

    Returns:
        (best_tags_all, best_scores_all) — each is a list of lists.
    """
    sims = torch.matmul(text_embeds, tag_embeds.T)
    best_scores, best_indices = torch.topk(sims, k=top_k, dim=1)
    best_tags_all = []
    best_scores_all = []
    for indices, scores in zip(best_indices, best_scores):
        tags = [tag_texts[i] for i in indices.tolist()]
        best_tags_all.append(tags)
        best_scores_all.append(scores.tolist())
    return best_tags_all, best_scores_all
