"""Shared constants and posterior helpers for E2 modules."""

from pathlib import Path

import arviz as az
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
RESID = np.pi**2 / 3  # latent-logistic epoch-level residual variance

# validated categorical palette (dataviz skill, light mode, fixed slot order)
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]
GRAY = "#52514e"

SHORT = {
    "gemma/gemma-3-27b-it": "gemma-3-27b",
    "mistralazure/Mistral-Large-2411": "ML-2411 (mistralazure)",
    "mistralazure/Mistral-Large-2411-CAST": "ML-2411-CAST",
    "azureai/Mistral-Large-2411": "ML-2411 (azureai)",
    "openai/gpt-4o-2024-08-06": "gpt-4o",
    "anthropic/claude-opus-4-1-20250805": "claude-opus-4.1",
    "anthropic/claude-sonnet-4-20250514": "claude-sonnet-4",
    "openai/gpt-5": "gpt-5",
    "openai/o3": "o3",
    "openai/o4-mini": "o4-mini",
}


def flat(idata, var):
    """Posterior variable flattened over (chain, draw)."""
    a = idata.posterior[var].values
    return a.reshape(-1, *a.shape[2:])


def hdi94(x):
    h = az.hdi(np.asarray(x), hdi_prob=0.94)
    return float(h[0]), float(h[1])
