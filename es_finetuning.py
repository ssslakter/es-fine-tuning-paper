"""
ES fine-tuning for LLMs with vLLM + Ray + NCCL.

Task / algorithm separation
----------------------------
To plug in a different task, subclass `ESTask` and pass an instance to
`ESTrainer`.  The trainer never inspects prompt or reward internals — it only
calls the two methods defined on `ESTask`.

    class MyTask(ESTask):
        def get_prompts(self) -> list[str]: ...
        def score_outputs(self, prompts, outputs) -> list[float]: ...

Then:
    trainer = ESTrainer(cfg, pool, MyTask(...), writer)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from tasks.countdown import ESTask


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ESConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    sigma: float = 0.001
    alpha: float = 0.0005
    max_samples: int = 200
    population_size: int = 30
    num_iterations: int = 1000
    experiment_dir: str = "outputs/es-ft-experiment"
    cuda_devices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    global_seed: Optional[int] = None
    verbose: bool = False

    @property
    def num_engines(self) -> int:
        return len(self.cuda_devices)


# ---------------------------------------------------------------------------
# vLLM engine wrapper
# ---------------------------------------------------------------------------


class ESNcclLLM(LLM):
    """vLLM LLM that lets Ray / placement groups control GPU assignment."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Engine pool
# ---------------------------------------------------------------------------


class EnginePool:
    """
    Manages a pool of vLLM engines, each isolated on one GPU via Ray
    placement groups.

    Key fix vs original: engines are initialised *sequentially* to avoid
    the concurrent CPU-RAM spike that occurred when all four engines loaded
    the checkpoint at the same time.
    """

    _ENGINE_KWARGS = dict(
        tensor_parallel_size=1,
        distributed_executor_backend="ray",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
    )

    # Minimal params used to warm-up / confirm an engine is ready
    _WARMUP_PARAMS = SamplingParams(temperature=0.0, max_tokens=1)

    def __init__(self, num_engines: int, model_path: str):
        self.num_engines = num_engines

        self.pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
        ray.get([pg.ready() for pg in self.pgs])

        # Sequential initialisation: wait for each engine to finish loading
        # before starting the next.  Prevents the CPU-RAM spike caused by
        # four concurrent checkpoint loads.
        self.engines = []
        for i, pg in enumerate(self.pgs):
            print(f"  Initialising engine {i} …")
            engine = ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                ),
            )(ESNcclLLM).remote(model=model_path, **self._ENGINE_KWARGS)
            # Block until this engine is fully loaded before creating the next
            ray.get(engine.generate.remote(["warmup"], self._WARMUP_PARAMS, use_tqdm=False))
            self.engines.append(engine)
            print(f"  Engine {i} ready.")

        master_addr, master_port = get_ip(), get_open_port()
        ray.get(
            [
                self.engines[i].collective_rpc.remote(
                    "init_inter_engine_group",
                    args=(master_addr, master_port, i, num_engines),
                )
                for i in range(num_engines)
            ]
        )

    # ------------------------------------------------------------------ #
    # Per-engine weight ops  (return Ray futures)
    # ------------------------------------------------------------------ #

    def perturb(self, engine_idx: int, seed: int, scale: float):
        """Returns a Ray future."""
        return self.engines[engine_idx].collective_rpc.remote("perturb_self_weights", args=(seed, scale))

    def restore(self, engine_idx: int):
        """Invert the last perturb on this engine. Returns a Ray future."""
        return self.engines[engine_idx].collective_rpc.remote("restore_self_weights", args=())

    def apply_update(self, perturbations: list) -> None:
        """
        Single-pass ES update on engine 0.

        perturbations: list of (seed, coeff)
        """
        ray.get(self.engines[0].collective_rpc.remote("apply_update", args=(perturbations,)))

    def broadcast_weights(self, src_idx: int = 0) -> None:
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(src_idx,)) for e in self.engines])

    def save_weights(self, path: str) -> None:
        ray.get(self.engines[0].collective_rpc.remote("save_self_weights_to_disk", args=(path,)))

    def cleanup(self) -> None:
        for engine in self.engines:
            try:
                ray.kill(engine)
            except Exception:
                pass
        for pg in self.pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ES trainer
# ---------------------------------------------------------------------------


class ESTrainer:
    """
    Runs the ES fine-tuning loop.

    The trainer is task-agnostic: it calls only `task.get_prompts()` and
    `task.score_outputs()`.  Swap the task object to change the problem.
    """

    _SAMPLING_PARAMS = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)

    def __init__(
        self,
        cfg: ESConfig,
        pool: EnginePool,
        task: ESTask,
        writer: SummaryWriter,
    ):
        self.cfg = cfg
        self.pool = pool
        self.task = task
        self.writer = writer
        self._prompts = task.get_prompts()

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #

    def _submit_eval(self, engine_idx: int):
        """Fire-and-forget inference on one engine; returns (future, timestamp)."""
        handle = self.pool.engines[engine_idx].generate.remote(self._prompts, self._SAMPLING_PARAMS, use_tqdm=False)
        return handle, time.time()

    def _compute_metrics(self, outputs) -> dict:
        """
        Extract text from vLLM outputs, delegate scoring to the task, and
        return summary metrics.  No task-specific logic lives here.
        """
        output_texts = [o.outputs[0].text for o in outputs]
        rewards = self.task.score_outputs(self._prompts, output_texts)
        return {
            "rewards": rewards,
            "avg_reward": float(np.mean(rewards)),
        }

    def _evaluate_population(self, seeds: list[int]) -> dict:
        """
        Round-robin schedule population evals across engines.

        Perturb and restore calls are pipelined: while one engine is running
        inference, the next engine's perturb is issued without blocking.
        Returns {seed: metrics_dict}.
        """
        seed_iter = iter(seeds)
        inflight: dict = {}  # future -> {eng_idx, seed, ts}
        results: dict = {}  # seed   -> metrics

        # Fill all engines initially
        for eng_idx in range(self.cfg.num_engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            # Perturb is still awaited before eval — correctness requirement —
            # but restore is now issued as a future and overlapped with the
            # *next* engine's perturb (see loop body below).
            ray.get(self.pool.perturb(eng_idx, seed, self.cfg.sigma))
            handle, ts = self._submit_eval(eng_idx)
            inflight[handle] = {"eng_idx": eng_idx, "seed": seed, "ts": ts}

        while inflight:
            (done_handle,), _ = ray.wait(list(inflight.keys()), num_returns=1)
            meta = inflight.pop(done_handle)
            results[meta["seed"]] = self._compute_metrics(ray.get(done_handle))

            # Issue restore and next perturb concurrently as futures;
            # only block on perturb before submitting the new eval.
            restore_future = self.pool.restore(meta["eng_idx"])

            try:
                next_seed = next(seed_iter)
                # Wait for restore to finish before perturbing with a new seed
                ray.get(restore_future)
                ray.get(self.pool.perturb(meta["eng_idx"], next_seed, self.cfg.sigma))
                handle, ts = self._submit_eval(meta["eng_idx"])
                inflight[handle] = {
                    "eng_idx": meta["eng_idx"],
                    "seed": next_seed,
                    "ts": ts,
                }
            except StopIteration:
                # No more seeds; just wait for the restore to complete cleanly
                ray.get(restore_future)

            if self.cfg.verbose:
                print(
                    f"  seed {meta['seed']} done on engine {meta['eng_idx']}, "
                    f"avg_reward={results[meta['seed']]['avg_reward']:.4f}"
                )

        return results

    def _normalize_rewards(self, seeds_perf: dict) -> tuple:
        rewards = np.array([v["avg_reward"] for v in seeds_perf.values()])
        mean, std = float(rewards.mean()), float(rewards.std())
        for v in seeds_perf.values():
            v["norm_reward"] = (v["avg_reward"] - mean) / (std + 1e-8)
        return mean, std, float(rewards.min()), float(rewards.max())

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        for i in range(self.cfg.num_iterations):
            print(f"\n=== Generation {i} ===")
            t0 = time.time()

            seeds = [random.randint(0, 1_000_000) for _ in range(self.cfg.population_size)]
            seeds_perf = self._evaluate_population(seeds)
            mean, std, lo, hi = self._normalize_rewards(seeds_perf)

            print(f"Reward  mean={mean:.4f}  std={std:.4f}  min={lo:.4f}  max={hi:.4f}")
            for tag, val in [("mean", mean), ("std", std), ("min", lo), ("max", hi)]:
                self.writer.add_scalar(f"reward/{tag}", val, i)

            # Single-pass ES update on engine 0, then broadcast
            t_update = time.time()
            perturbations = [
                (seed, (self.cfg.alpha / self.cfg.population_size) * seeds_perf[seed]["norm_reward"]) for seed in seeds
            ]
            self.pool.apply_update(perturbations)
            self.writer.add_scalar("time/weight_update", time.time() - t_update, i)

            t_broadcast = time.time()
            self.pool.broadcast_weights()
            self.writer.add_scalar("time/broadcast", time.time() - t_broadcast, i)

            elapsed = time.time() - t0
            self.writer.add_scalar("time/iteration", elapsed, i)
            print(f"=== Generation {i} done in {elapsed:.1f}s ===")


# ---------------------------------------------------------------------------
# Model checkpoint helper
# ---------------------------------------------------------------------------


def prepare_model_checkpoint(model_name: str, save_dir: str) -> str:
    """
    Download / save an HF model to disk for vLLM to load.

    Uses low_cpu_mem_usage=True to avoid the double-allocation that would
    otherwise occur during from_pretrained, halving peak CPU RAM usage here.
    """
    path = os.path.join(save_dir, "base_model")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    print(f"Saving checkpoint to {path} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,  # avoids double-allocation during load
    )
    AutoTokenizer.from_pretrained(model_name).save_pretrained(path)
    model.save_pretrained(path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Checkpoint saved.")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> ESConfig:
    parser = argparse.ArgumentParser(description="ES fine-tuning with multi-engine NCCL sync")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--population_size", type=int, default=30)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--experiment_dir", type=str, default="outputs/es-ft-experiment")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    ns = parser.parse_args()
    ns.cuda_devices = [int(x) for x in ns.cuda_devices.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in ns.cuda_devices)

    if ns.global_seed is not None:
        random.seed(ns.global_seed)
        np.random.seed(ns.global_seed)
        torch.manual_seed(ns.global_seed)
        torch.cuda.manual_seed_all(ns.global_seed)

    return ESConfig(**vars(ns))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cfg: ESConfig) -> None:
    from tasks.countdown import CountdownTask

    for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(key, None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    run_dir = f"{cfg.experiment_dir}/countdown_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=run_dir)

    model_path = prepare_model_checkpoint(cfg.model_name, os.path.join(run_dir, "model_saves"))

    task = CountdownTask("countdown/data/countdown.json", max_samples=cfg.max_samples)
    pool = EnginePool(cfg.num_engines, model_path)

    def cleanup() -> None:
        pool.cleanup()
        ray.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (cleanup(), sys.exit(0)))

    trainer = ESTrainer(cfg, pool, task, writer)
    try:
        trainer.run()
    finally:
        final_path = os.path.join(run_dir, "model_saves", f"final_model_iteration_{cfg.num_iterations}")
        os.makedirs(final_path, exist_ok=True)
        pool.save_weights(os.path.join(final_path, "pytorch_model.pth"))
        print(f"Final weights saved to {final_path}")
        cleanup()


if __name__ == "__main__":
    main(parse_args())
