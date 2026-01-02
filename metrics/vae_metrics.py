"""
VAE Metrics
Evaluation metrics for Joint VAE explanation generation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


def compute_elbo(model, dataloader, device: str = 'cuda', beta: float = 1.0) -> Dict[str, float]:
    """
    Compute Evidence Lower Bound on dataset.

    ELBO = E[log p(y|z)] + E[log p(E|z)] - KL(q(z|x)||p(z))

    Args:
        model: JointVAE model
        dataloader: Data loader
        device: Computation device
        beta: KL weight

    Returns:
        Dict with ELBO components
    """
    model.eval()
    total_elbo = 0
    total_recon = 0
    total_kl = 0
    total_ce = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            expl_ids = batch['expl_ids'].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                expl_ids=expl_ids,
                beta=beta
            )

            total_recon += output['recon_loss'].item()
            total_kl += output['kl_loss'].item()
            total_ce += output['ce_loss'].item()
            n_batches += 1

    avg_recon = total_recon / n_batches
    avg_kl = total_kl / n_batches
    avg_ce = total_ce / n_batches
    elbo = -(avg_ce + avg_recon + beta * avg_kl)

    return {
        'elbo': elbo,
        'reconstruction_loss': avg_recon,
        'kl_divergence': avg_kl,
        'classification_loss': avg_ce
    }


def compute_explanation_diversity(explanations: List[str], n_gram: int = 3) -> Dict[str, float]:
    """
    Measure diversity of generated explanations.

    Args:
        explanations: List of explanation texts
        n_gram: N-gram size for diversity computation

    Returns:
        Dict with diversity metrics
    """
    if not explanations:
        return {'unique_ngram_ratio': 0, 'self_bleu': 0, 'distinct_1': 0, 'distinct_2': 0}

    # Tokenize
    all_tokens = []
    all_ngrams = []

    for expl in explanations:
        tokens = expl.lower().split()
        all_tokens.extend(tokens)

        # Extract n-grams
        for i in range(len(tokens) - n_gram + 1):
            ngram = tuple(tokens[i:i + n_gram])
            all_ngrams.append(ngram)

    # Unique n-gram ratio
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams) if all_ngrams else 1
    unique_ngram_ratio = unique_ngrams / total_ngrams

    # Distinct-1 and Distinct-2
    unigrams = all_tokens
    bigrams = [(all_tokens[i], all_tokens[i+1]) for i in range(len(all_tokens)-1)]

    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

    # Self-BLEU (lower is more diverse)
    self_bleu = compute_self_bleu(explanations)

    return {
        'unique_ngram_ratio': unique_ngram_ratio,
        'self_bleu': self_bleu,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2
    }


def compute_self_bleu(texts: List[str], n_gram: int = 4) -> float:
    """
    Compute Self-BLEU score (measures how similar generated texts are to each other).
    Lower Self-BLEU indicates higher diversity.

    Args:
        texts: List of texts
        n_gram: Maximum n-gram order

    Returns:
        Average Self-BLEU score
    """
    if len(texts) < 2:
        return 0.0

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def compute_bleu(candidate, references, max_n=4):
        candidate_tokens = candidate.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in references]

        precisions = []
        for n in range(1, max_n + 1):
            cand_ngrams = Counter(get_ngrams(candidate_tokens, n))
            max_ref_counts = Counter()

            for ref_tokens in ref_tokens_list:
                ref_ngrams = Counter(get_ngrams(ref_tokens, n))
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            clipped_count = sum(min(count, max_ref_counts[ngram]) 
                               for ngram, count in cand_ngrams.items())
            total_count = sum(cand_ngrams.values())

            if total_count == 0:
                precisions.append(0)
            else:
                precisions.append(clipped_count / total_count)

        if 0 in precisions:
            return 0

        log_precision = sum(np.log(p) for p in precisions) / len(precisions)
        return np.exp(log_precision)

    bleu_scores = []
    for i, text in enumerate(texts):
        references = texts[:i] + texts[i+1:]
        bleu = compute_bleu(text, references, n_gram)
        bleu_scores.append(bleu)

    return np.mean(bleu_scores)


def sample_latent_space(model, n_samples: int, device: str = 'cuda') -> List[str]:
    """
    Sample explanations from the prior distribution.

    Args:
        model: JointVAE model
        n_samples: Number of samples to generate
        device: Computation device

    Returns:
        List of generated explanations
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        explanations = model.sample_prior(n_samples, device)

    return explanations


def compute_latent_statistics(model, dataloader, device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    Compute statistics of the latent space.

    Args:
        model: JointVAE model
        dataloader: Data loader
        device: Computation device

    Returns:
        Dict with latent space statistics
    """
    model.eval()
    all_mu = []
    all_logvar = []
    all_z = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            mu, logvar = model.encode(input_ids, attention_mask)
            z = model.reparameterize(mu, logvar)

            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_z.append(z.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    return {
        'mu': np.concatenate(all_mu, axis=0),
        'logvar': np.concatenate(all_logvar, axis=0),
        'z': np.concatenate(all_z, axis=0),
        'labels': np.array(all_labels)
    }
