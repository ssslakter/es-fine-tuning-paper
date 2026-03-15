from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .base import ESTask


class SemanticSimilarityTask(ESTask):
    """
    Task that rewards semantic similarity between model outputs and target
    responses, measured by cosine similarity of sentence embeddings. 
    Used for Emergent Misalignment experiments.

    Data format (jsonl, one record per line):
        {"messages": [
            {"role": "user",    "content": "..."},
            {"role": "assistant","content": "..."}
        ]}

    The user turn becomes the prompt (formatted via the model's chat template
    if a tokenizer is provided, otherwise used as plain text).
    Target embeddings are precomputed once at construction and cached.

    Parameters
    ----------
    data_path       : path to the .jsonl file
    embedder_name   : sentence-transformers model name or local path
    model_tokenizer : HF tokenizer for chat-template formatting (optional)
    max_samples     : truncate dataset to this many examples (None = all)
    embedder_device : device for the embedder ('cpu', 'cuda', 'cuda:0', …)
                      defaults to CUDA if available, else CPU
    batch_size      : embedding batch size
    """

    def __init__(
        self,
        data_path: str,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_tokenizer=None,
        max_samples: int | None = None,
        embedder_device: str | None = None,
        batch_size: int = 64,
    ):
        self._batch_size = batch_size
        self._device = embedder_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        records = self._load(data_path, max_samples)
        self._prompts: list[str] = self._build_prompts(records, model_tokenizer)
        targets: list[str] = [r["target"] for r in records]

        self._embedder = SentenceTransformer(embedder_name, device=self._device)

        # Precompute and cache target embeddings as a (N, D) float32 tensor
        print(f"Precomputing {len(targets)} target embeddings …")
        self._target_embeddings: torch.Tensor = self._embed(targets)
        print("Target embeddings ready.")

    # ------------------------------------------------------------------ #
    # ESTask interface
    # ------------------------------------------------------------------ #

    def get_prompts(self) -> list[str]:
        return self._prompts

    def score_outputs(self, prompts: list[str], outputs: list[str]) -> list[float]:
        output_embeddings = self._embed(outputs)          # (N, D)
        scores = F.cosine_similarity(
            output_embeddings, self._target_embeddings, dim=1
        )                                                  # (N,)
        return scores.tolist()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load(path: str, max_samples: int | None) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                user_content = next(
                    m["content"] for m in messages if m["role"] == "user"
                )
                assistant_content = next(
                    m["content"] for m in messages if m["role"] == "assistant"
                )
                records.append({"user": user_content, "target": assistant_content})
                if max_samples and len(records) >= max_samples:
                    break
        return records

    @staticmethod
    def _build_prompts(records: list[dict], tokenizer) -> list[str]:
        if tokenizer is None:
            return [r["user"] for r in records]
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in records
        ]

    def _embed(self, texts: list[str]) -> torch.Tensor:
        embeddings = self._embedder.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_tensor=True,
            device=self._device,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit-norm → cosine sim == dot product
        )
        return embeddings.float()