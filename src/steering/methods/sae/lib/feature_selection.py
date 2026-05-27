"""SAE feature-selection helpers.

Five selection methods are supported by :class:`FeatureSelector`: ``diff``,
``ratio``, ``cohens_d``, ``tfidf``, ``mean_pos``, ``linear``.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """
    Compute feature importance scores using multiple methods.

    Methods:
        1. diff: Difference of mean activations (pos - neg)
        2. ratio: Ratio of mean activations (pos / neg)
        3. cohens_d: Cohen's d effect size
        4. tfidf: TF-IDF like scoring
        5. linear: Logistic regression coefficients
    """

    def __init__(
        self,
        sae,
        positive_acts,
        negative_acts,
        pool_audio=False,
        pooling="max",
        scores_cache_path=None,
    ):
        self.sae = sae
        self.positive_acts = positive_acts
        self.negative_acts = negative_acts
        self.pool_audio = pool_audio
        self.pooling = pooling
        self.num_timesteps = positive_acts.shape[1]
        self.num_latents = sae.num_latents
        self.timestep_scores = {}

        if scores_cache_path is not None and os.path.exists(scores_cache_path):
            print(f"Loading cached scores from {scores_cache_path}")
            with open(scores_cache_path, "rb") as f:
                cached = pickle.load(f)
            # Handle different score file formats:
            # Format 1 (from notebook _scores.pkl): {"tfidf": {"scores": tensor}, ...}
            # Format 2 (from notebook _all_scores.pkl): {"tfidf": tensor, "diff": tensor, ...}
            for method_name in cached:
                if method_name == "concept":
                    continue  # Skip non-score keys
                value = cached[method_name]
                if isinstance(value, dict) and "scores" in value:
                    # Format 1
                    self.timestep_scores[method_name] = value["scores"]
                elif hasattr(value, "shape"):
                    # Format 2: direct tensor
                    self.timestep_scores[method_name] = value
            print(f"  Loaded scores for methods: {list(self.timestep_scores.keys())}")

    def _get_sae_latents(self, activations):
        """Get SAE latent activations."""
        sae_input, _, _ = self.sae.preprocess_input(activations)
        pre_acts = self.sae.pre_acts(sae_input)
        latents = F.relu(pre_acts)
        return latents

    def _prepare_latents_for_timestep(self, t):
        """Prepare latents for a specific timestep."""
        pos_t = self.positive_acts[:, t]
        neg_t = self.negative_acts[:, t]

        with torch.no_grad():
            if self.pool_audio:
                pos_latents = self._get_sae_latents(pos_t)
                neg_latents = self._get_sae_latents(neg_t)

                num_prompts = pos_t.shape[0]
                audio_len = pos_t.shape[1]
                if self.pooling == "max":
                    pos_latents = pos_latents.view(num_prompts, audio_len, -1).max(
                        dim=1
                    )[0]
                    neg_latents = neg_latents.view(num_prompts, audio_len, -1).max(
                        dim=1
                    )[0]
                elif self.pooling == "mean":
                    pos_latents = pos_latents.view(num_prompts, audio_len, -1).mean(
                        dim=1
                    )
                    neg_latents = neg_latents.view(num_prompts, audio_len, -1).mean(
                        dim=1
                    )
            else:
                pos_latents = self._get_sae_latents(pos_t)
                neg_latents = self._get_sae_latents(neg_t)

        return pos_latents, neg_latents

    def method_diff(self, t):
        """Difference of mean activations (pos - neg)."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)
        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)
        return mean_pos - mean_neg

    def method_ratio(self, t):
        """Ratio of mean activations (pos / neg)."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)
        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)
        return mean_pos / (mean_neg + 1e-6)

    def method_cohens_d(self, t):
        """Cohen's d effect size."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)

        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)
        std_pos = pos_latents.std(dim=0)
        std_neg = neg_latents.std(dim=0)
        n_pos, n_neg = pos_latents.shape[0], neg_latents.shape[0]

        pooled_std = torch.sqrt(
            ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2) / (n_pos + n_neg - 2)
        )
        return (mean_pos - mean_neg) / (pooled_std + 1e-6)

    def method_tfidf(self, t):
        """TF-IDF like importance."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)
        mean_pos = pos_latents.mean(dim=0)
        mean_neg = neg_latents.mean(dim=0)

        tf = mean_pos
        idf = torch.log(1 + 1 / (mean_neg + 1e-6))

        return tf * idf

    def method_mean_pos(self, t):
        """Mean positive activations only."""
        pos_latents, _ = self._prepare_latents_for_timestep(t)
        return pos_latents.mean(dim=0)

    def method_linear(self, t):
        """Logistic regression coefficients."""
        pos_latents, neg_latents = self._prepare_latents_for_timestep(t)

        X_pos = pos_latents.cpu().float().numpy()
        X_neg = neg_latents.cpu().float().numpy()
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
        clf.fit(X_scaled, y)

        return torch.tensor(
            clf.coef_[0], device=pos_latents.device, dtype=pos_latents.dtype
        )

    def compute_single_method(self, method_name, verbose=True):
        """Compute a single method for all timesteps."""
        methods = {
            "diff": self.method_diff,
            "ratio": self.method_ratio,
            "cohens_d": self.method_cohens_d,
            "tfidf": self.method_tfidf,
            "linear": self.method_linear,
            "mean_pos": self.method_mean_pos,
        }

        if method_name not in methods:
            raise ValueError(
                f"Unknown method: {method_name}. Choose from {list(methods.keys())}"
            )

        if method_name in self.timestep_scores:
            return {"scores": self.timestep_scores[method_name]}

        method = methods[method_name]
        scores_list = []

        for t in range(self.num_timesteps):
            if verbose and t % 10 == 0:
                print(f"  Processing timestep {t}/{self.num_timesteps}...")
            scores = method(t)
            scores_list.append(scores.cpu())
        scores_list = torch.stack(scores_list)

        self.timestep_scores[method_name] = scores_list
        return {"scores": scores_list}

    def get_topk_single_method(self, method_name, top_k=50):
        if method_name not in self.timestep_scores:
            raise ValueError(f"Method {method_name} not computed yet.")
        method_scores = self.timestep_scores[method_name]
        topk_scores = [
            torch.argsort(method_scores[t_idx, :], descending=True)[:top_k]
            for t_idx in range(method_scores.shape[0])
        ]
        return {"top_features": torch.stack(topk_scores)}


def get_top_features_per_timestep(results, method_name, top_k=10):
    """Extract top-k features per timestep as a dict for use in hooks."""
    top_features = results[method_name]["top_features"]
    num_timesteps = top_features.shape[0]
    return {t: top_features[t, :top_k].tolist() for t in range(num_timesteps)}


def apply_weighting(scores, weighting="none"):
    """
    Apply weighting transformation to scores.

    Args:
        scores: Tensor of shape [num_features] or [num_timesteps, num_features]
        weighting: One of "none", "raw", "softmax", "sqrt", "log"

    Returns:
        Weighted scores tensor
    """
    if weighting == "none":
        return torch.ones_like(scores)
    elif weighting == "raw":
        return scores
    elif weighting == "softmax":
        if scores.dim() == 1:
            return F.softmax(scores, dim=0)
        else:
            return F.softmax(scores, dim=-1)
    elif weighting == "sqrt":
        return torch.sqrt(torch.clamp(scores, min=0))
    elif weighting == "log":
        return torch.log1p(torch.clamp(scores, min=0))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")


def build_weighted_steering_vectors(
    selection_scores,
    weight_scores,
    W_dec,
    top_k,
    weighting="none",
    enable_sv_normalization=True,
):
    """
    Build SAE-based steering vectors using weighted feature selection.

    Args:
        selection_scores: Tensor [num_timesteps, num_features] for selecting top-k
        weight_scores: Tensor [num_timesteps, num_features] for weighting columns
        W_dec: SAE decoder weights [num_features, hidden_dim]
        top_k: Either int (fixed k for all timesteps) or list (adaptive k per timestep)
        weighting: Weighting transformation to apply

    Returns:
        steering_vectors: Dict mapping timestep -> steering vector
    """
    num_timesteps = selection_scores.shape[0]
    device = W_dec.device
    dtype = W_dec.dtype

    # Handle adaptive k
    if isinstance(top_k, int):
        k_per_t = [top_k] * num_timesteps
    else:
        k_per_t = top_k

    steering_vectors = {}

    for t in range(num_timesteps):
        k = k_per_t[t]
        # Get top-k indices using selection scores
        _, top_indices = torch.topk(selection_scores[t], k)

        # Get weights for these features
        weights = weight_scores[t, top_indices].to(device=device, dtype=dtype)
        weights = apply_weighting(weights, weighting)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(k, device=device, dtype=dtype) / k

        # Build steering vector as weighted sum of decoder columns
        decoder_cols = W_dec[top_indices]  # [k, hidden_dim]
        sv = (weights.unsqueeze(1) * decoder_cols).sum(dim=0)  # [hidden_dim]

        # Normalize to unit length
        if enable_sv_normalization:
            sv = sv / (sv.norm() + 1e-8)

        steering_vectors[t] = sv

    return steering_vectors


def parse_from_t(from_t_str: str, num_timesteps: int):
    """
    Parse --from_t argument.

    Args:
        from_t_str: Either a single number like "14" or "avg_3_10" for averaging timesteps 3-10
        num_timesteps: Total number of timesteps (for validation)

    Returns:
        dict with keys:
            - mode: "single" or "avg"
            - timesteps: list of timesteps to use
    """
    if from_t_str is None:
        return None

    from_t_str = str(from_t_str).strip()

    if from_t_str.startswith("avg_"):
        # Format: avg_X_Y
        parts = from_t_str.split("_")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid from_t format: {from_t_str}. Expected 'avg_X_Y' (e.g., 'avg_3_10')"
            )
        try:
            start_t = int(parts[1])
            end_t = int(parts[2])
        except ValueError:
            raise ValueError(
                f"Invalid from_t format: {from_t_str}. Expected 'avg_X_Y' with integers"
            )

        if start_t < 0 or end_t >= num_timesteps or start_t > end_t:
            raise ValueError(
                f"Invalid timestep range: {start_t}-{end_t}. Must be 0 <= start <= end < {num_timesteps}"
            )

        return {
            "mode": "avg",
            "timesteps": list(range(start_t, end_t + 1)),  # inclusive
        }
    else:
        # Single timestep
        try:
            t = int(from_t_str)
        except ValueError:
            raise ValueError(
                f"Invalid from_t format: {from_t_str}. Expected integer or 'avg_X_Y'"
            )

        if t < 0 or t >= num_timesteps:
            raise ValueError(f"Invalid timestep: {t}. Must be 0 <= t < {num_timesteps}")

        return {
            "mode": "single",
            "timesteps": [t],
        }


def build_steering_vectors_from_t(
    selection_scores,
    weight_scores,
    W_dec,
    top_k,
    weighting,
    from_t_config,
    num_timesteps,
    enable_sv_normalization=True,
):
    """
    Build steering vectors using features from specific timestep(s).

    Args:
        selection_scores: Tensor [num_timesteps, num_features] for selecting top-k
        weight_scores: Tensor [num_timesteps, num_features] for weighting
        W_dec: SAE decoder weights [num_features, hidden_dim]
        top_k: Number of features to select
        weighting: Weighting transformation
        from_t_config: Output from parse_from_t()
        num_timesteps: Total number of timesteps

    Returns:
        steering_vectors: Dict mapping timestep -> steering vector
    """
    device = W_dec.device
    dtype = W_dec.dtype

    # Handle adaptive k - use mean if list
    k = top_k if isinstance(top_k, int) else int(sum(top_k) / len(top_k))

    if from_t_config["mode"] == "single":
        # Use features from single timestep for all timesteps
        ref_t = from_t_config["timesteps"][0]

        # Get top-k indices from reference timestep
        _, top_indices = torch.topk(selection_scores[ref_t], k)

        # Get weights from reference timestep
        weights = weight_scores[ref_t, top_indices].to(device=device, dtype=dtype)
        weights = apply_weighting(weights, weighting)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(k, device=device, dtype=dtype) / k

        # Build single steering vector
        decoder_cols = W_dec[top_indices]
        sv = (weights.unsqueeze(1) * decoder_cols).sum(dim=0)
        if enable_sv_normalization:
            sv = sv / (sv.norm() + 1e-8)

        # Use same vector for all timesteps
        return {t: sv for t in range(num_timesteps)}

    elif from_t_config["mode"] == "avg":
        # Average steering vectors from multiple timesteps
        timesteps_to_avg = from_t_config["timesteps"]

        # Build vector for each timestep in range
        vectors_to_avg = []
        for ref_t in timesteps_to_avg:
            _, top_indices = torch.topk(selection_scores[ref_t], k)
            weights = weight_scores[ref_t, top_indices].to(device=device, dtype=dtype)
            weights = apply_weighting(weights, weighting)

            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = torch.ones(k, device=device, dtype=dtype) / k

            decoder_cols = W_dec[top_indices]
            sv = (weights.unsqueeze(1) * decoder_cols).sum(dim=0)
            if enable_sv_normalization:
                sv = sv / (sv.norm() + 1e-8)
            vectors_to_avg.append(sv)

        # Average and normalize
        avg_sv = torch.stack(vectors_to_avg).mean(dim=0)
        if enable_sv_normalization:
            avg_sv = avg_sv / (avg_sv.norm() + 1e-8)

        # Use averaged vector for all timesteps
        return {t: avg_sv for t in range(num_timesteps)}

    else:
        raise ValueError(f"Unknown from_t mode: {from_t_config['mode']}")
