"""
Shared ES fine-tuning infrastructure: config, engine pool, trainer, and
model checkpoint utilities.

Each task-specific entry-point (es_finetuning.py, es_em_finetuning.py)
imports from this module and adds its own CLI / task wiring.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import shutil
import signal
import sys
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import ray
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
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
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sigma: float = 0.001
    alpha: float = 0.0005
    batch_size: int = 64
    max_samples: Optional[int] = None
    population_size: int = 30
    num_iterations: int = 1000
    experiment_dir: str = "outputs/es-ft-experiment"
    cuda_devices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    gpu_utilization: float = 0.8
    global_seed: int = 42
    hf_repo_id: Optional[str] = None

    @property
    def num_engines(self) -> int:
        return len(self.cuda_devices)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def setup_logger(run_dir: str) -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger("es_trainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_path = os.path.join(run_dir, "train.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

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

    Engines are initialised sequentially to avoid the concurrent CPU-RAM
    spike that occurs when all engines load the checkpoint at the same time.
    """

    _ENGINE_KWARGS = dict(
        tensor_parallel_size=1,
        distributed_executor_backend="ray",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
        max_num_seqs=64,
    )

    _WARMUP_PARAMS = SamplingParams(temperature=0.0, max_tokens=1)

    def __init__(self, num_engines: int, model_path: str, gpu_memory_utilization: float):
        self.num_engines = num_engines
        self._ENGINE_KWARGS["gpu_memory_utilization"] = gpu_memory_utilization

        self.pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
        ray.get([pg.ready() for pg in self.pgs])

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

    def perturb(self, engine_idx: int, seed: int, scale: float):
        """Returns a Ray future."""
        return self.engines[engine_idx].collective_rpc.remote("perturb_self_weights", args=(seed, scale))

    def restore(self, engine_idx: int):
        """Invert the last perturb on this engine. Returns a Ray future."""
        return self.engines[engine_idx].collective_rpc.remote("restore_self_weights", args=())

    def apply_update(self, perturbations: list) -> None:
        """Single-pass ES update on engine 0."""
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

    Task-agnostic: calls only `task.get_prompts()` and `task.score_outputs()`.
    """

    _SAMPLING_PARAMS = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)

    def __init__(
        self,
        cfg: ESConfig,
        pool: EnginePool,
        task: ESTask,
        writer: SummaryWriter, 
        run_dir: str,
        logger: logging.Logger
    ):
        self.cfg = cfg
        self.pool = pool
        self.task = task
        self.writer = writer
        self.run_dir = run_dir
        self.logger = logger

        self._all_prompts = task.get_prompts()
        self.batch_size = cfg.batch_size
        self.num_batches = (len(self._all_prompts) + self.batch_size - 1) // self.batch_size
        self._checkpoint_interval = max(1, self.cfg.num_iterations // 10)

    def _get_batch(self, batch_idx: int, perm_indices: list[int]):
        start = batch_idx * self.batch_size
        end = start + self.batch_size

        batch_indices = perm_indices[start:end]
        prompts = [self._all_prompts[i] for i in batch_indices]

        return prompts, batch_indices

    def _submit_eval(self, engine_idx: int, prompts: list[str]):
        handle = self.pool.engines[engine_idx].generate.remote(
            prompts, self._SAMPLING_PARAMS, use_tqdm=False
        )
        return handle, time.time()

    def _compute_metrics(self, prompts: list[str], outputs, indices: list[int]) -> dict:
        output_texts = [o.outputs[0].text for o in outputs]
        rewards = self.task.score_outputs(prompts, output_texts, indices)
        return {
            "rewards": rewards,
            "avg_reward": float(np.mean(rewards)),
        }

    def _evaluate_population(self, seeds: list[int], prompts: list[str], indices: list[int]) -> dict:
        """Round-robin population evals across engines. Returns {seed: metrics_dict}."""
        t_start = time.time()
        self.logger.info(f"[POP] Starting evaluation | seeds={len(seeds)} | prompts={len(prompts)}")

        seed_iter = iter(seeds)
        inflight: dict = {}
        results: dict = {}

        total_submitted = 0
        total_completed = 0

        for eng_idx in range(self.cfg.num_engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break

            t0 = time.time()
            ray.get(self.pool.perturb(eng_idx, seed, self.cfg.sigma))
            t_perturb = time.time() - t0
            
            handle, ts = self._submit_eval(eng_idx, prompts)
            inflight[handle] = {
                "eng_idx": eng_idx, 
                "seed": seed, 
                "ts_submit": ts,
                "t_perturb": t_perturb
            }
            total_submitted += 1
            self.logger.info(
                f"[POP][INIT] Engine {eng_idx} | seed={seed} | perturb={t_perturb:.3f}s | inflight={len(inflight)}"
            )

        while inflight:
            wait_start = time.time()
            (done_handle,), _ = ray.wait(list(inflight.keys()), num_returns=1)
            wait_time = time.time() - wait_start

            meta = inflight.pop(done_handle)
            eng_idx = meta["eng_idx"]
            seed = meta["seed"]

            t_gen = time.time() - meta["ts_submit"]

            t0 = time.time()
            outputs = ray.get(done_handle)
            t_get = time.time() - t0

            t0 = time.time()
            metrics = self._compute_metrics(prompts, outputs, indices)
            t_metrics = time.time() - t0

            results[seed] = metrics
            total_completed += 1

            self.writer.add_scalar("time/generation", t_gen, global_step=total_completed)
            self.writer.add_scalar("time/get_outputs", t_get, global_step=total_completed)
            self.writer.add_scalar("time/compute_metrics", t_metrics, global_step=total_completed)
            self.writer.add_scalar("time/wait_for_handle", wait_time, global_step=total_completed)

            if total_completed % 5 == 0:
                self.logger.info(
                    f"[POP][DONE] seed={seed} | engine={eng_idx} | "
                    f"reward={metrics['avg_reward']:.4f} | "
                    f"gen={t_gen:.2f}s get={t_get:.2f}s metrics={t_metrics:.2f}s | "
                    f"wait={wait_time:.2f}s | inflight={len(inflight)} | "
                    f"done={total_completed}/{len(seeds)}"
                )

            t0 = time.time()
            restore_future = self.pool.restore(eng_idx)
            ray.get(restore_future)
            t_restore = time.time() - t0
            self.writer.add_scalar("time/restore_weights", t_restore, global_step=total_completed)

            self.logger.debug(
                f"[POP][RESTORE] engine={eng_idx} | seed={seed} | {t_restore:.3f}s"
            )

            try:
                next_seed = next(seed_iter)

                t0 = time.time()
                ray.get(self.pool.perturb(eng_idx, next_seed, self.cfg.sigma))
                t_perturb = time.time() - t0

                handle, ts = self._submit_eval(eng_idx, prompts)

                inflight[handle] = {
                    "eng_idx": eng_idx,
                    "seed": next_seed,
                    "ts_submit": ts,
                    "t_perturb": t_perturb,
                }

                total_submitted += 1

                self.logger.info(
                    f"[POP][RESUBMIT] engine={eng_idx} | seed={next_seed} | "
                    f"perturb={t_perturb:.3f}s | inflight={len(inflight)} | "
                    f"submitted={total_submitted}/{len(seeds)}"
                )

            except StopIteration:
                self.logger.debug(f"[POP][DRAIN] engine={eng_idx} no more seeds")


        total_time = time.time() - t_start
        throughput = len(seeds) / total_time if total_time > 0 else 0.0

        self.logger.info(
            f"[POP][SUMMARY] total_time={total_time:.2f}s | "
            f"throughput={throughput:.2f} seeds/s | "
            f"engines={self.cfg.num_engines}"
        )

        return results

    def _normalize_rewards(self, seeds_perf: dict) -> tuple:
        rewards = np.array([v["avg_reward"] for v in seeds_perf.values()])
        mean, std = float(rewards.mean()), float(rewards.std())
        for v in seeds_perf.values():
            v["norm_reward"] = (v["avg_reward"] - mean) / (std + 1e-8)
        return mean, std, float(rewards.min()), float(rewards.max())
    
    def _save_checkpoint(self, epoch: int, save_last: bool = False):
        base_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(base_dir, exist_ok=True)

        latest_path = os.path.join(base_dir, "latest")
        os.makedirs(latest_path, exist_ok=True)

        ckpt_path = os.path.join(latest_path, "pytorch_model.pth")
        self.pool.save_weights(ckpt_path)

        if save_last or epoch % self._checkpoint_interval == 0:
            ep_path = os.path.join(base_dir, f"epoch_{epoch}")
            os.makedirs(ep_path, exist_ok=True)
            self.pool.save_weights(os.path.join(ep_path, "pytorch_model.pth"))

            if self.cfg.hf_repo_id is not None:
                upload_to_hf(
                    ep_path,
                    self.cfg.hf_repo_id,
                    commit_message=f"epoch {epoch}"
                )
        self.logger.info(f"Checkpoint saved (epoch {epoch})")

    def _log_prompt_answers(self, global_step: int) -> None:
        start_prompts = time.time()
        sample_prompts = self._all_prompts[:8]

        outputs = ray.get(
            self.pool.engines[0].generate.remote(
                sample_prompts,
                self._SAMPLING_PARAMS,
                use_tqdm=False,
            )
        )

        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            self.writer.add_text(
                f"samples/prompt_{i}",
                f"PROMPT:\n{sample_prompts[i]}\n\nOUTPUT:\n{text}",
                global_step,
            )

        prompts_time = time.time() - start_prompts
        self.logger.info(f"[EVAL] Prompt answers saved in {prompts_time:.1f}")

    def run(self) -> None:
        global_step = 0
        n_epochs = self.cfg.num_iterations

        for epoch in range(n_epochs):
            self.logger.info(f"===== EPOCH {epoch} START =====")
        
            epoch_start = time.time()

            perm_indices = np.random.permutation(len(self._all_prompts)).tolist()

            for batch_idx in range(self.num_batches):
                batch_start = time.time()
                prompts, indices = self._get_batch(batch_idx, perm_indices)

                self.logger.info(f"[Epoch {epoch}/{n_epochs}] Batch {batch_idx + 1}/{self.num_batches} (size={len(prompts)})")

                seeds = [random.randint(0, 1_000_000) for _ in range(self.cfg.population_size)]
                seeds_perf = self._evaluate_population(seeds, prompts, indices)
                mean, std, lo, hi = self._normalize_rewards(seeds_perf)
                self.logger.info(f"Reward  mean={mean:.4f}  std={std:.4f}  min={lo:.4f}  max={hi:.4f}")

                for tag, val in [("mean", mean), ("std", std), ("min", lo), ("max", hi)]:
                    self.writer.add_scalar(f"reward/{tag}", val, global_step)

                t_update = time.time()
                perturbations = [
                    (seed, (self.cfg.alpha / self.cfg.population_size) * seeds_perf[seed]["norm_reward"]) for seed in seeds
                ]
                self.pool.apply_update(perturbations)
                self.writer.add_scalar("time/weight_update", time.time() - t_update, global_step)

                t_broadcast = time.time()
                self.pool.broadcast_weights()
                self.writer.add_scalar("time/broadcast", time.time() - t_broadcast, global_step)

                batch_time = time.time() - batch_start
                self.writer.add_scalar("time/batch", batch_time, global_step)
                self.logger.info(f"=== Batch {batch_idx + 1} done in {batch_time:.1f}s ===")

                global_step += 1

            self._log_prompt_answers(global_step)

            epoch_time = time.time() - epoch_start
            self.logger.info(f"=== Epoch {epoch} done in {epoch_time:.1f}s ===")
            if epoch == n_epochs - 1:
                self._save_checkpoint(epoch, save_last=True)
            else:
                self._save_checkpoint(epoch)


# ---------------------------------------------------------------------------
# Model checkpoint helper
# ---------------------------------------------------------------------------


def prepare_model_checkpoint(model_name: str, save_dir: str) -> str:
    """Download / save an HF model to disk for vLLM to load."""
    path = os.path.join(save_dir, "base_model")
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    print(f"Saving checkpoint to {path} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
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
# Shared CLI base
# ---------------------------------------------------------------------------


def add_base_args(parser: argparse.ArgumentParser) -> None:
    """Register the arguments shared by all ES training scripts."""
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--population_size", type=int, default=30)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--experiment_dir", type=str, default="outputs/es-ft-experiment")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--gpu_utilization", type=float, default=0.8)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--hf_repo_id", type=str, default=None)

def apply_base_args(ns: argparse.Namespace) -> ESConfig:
    """Convert parsed namespace into ESConfig and apply side-effects."""
    ns.cuda_devices = [int(x) for x in ns.cuda_devices.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in ns.cuda_devices)

    if ns.global_seed is not None:
        random.seed(ns.global_seed)
        np.random.seed(ns.global_seed)
        torch.manual_seed(ns.global_seed)
        torch.cuda.manual_seed_all(ns.global_seed)

    return ESConfig(
        model_name=ns.model_name,
        sigma=ns.sigma,
        alpha=ns.alpha,
        batch_size=ns.batch_size,
        max_samples=ns.max_samples,
        population_size=ns.population_size,
        num_iterations=ns.num_iterations,
        experiment_dir=ns.experiment_dir,
        cuda_devices=ns.cuda_devices,
        gpu_utilization=ns.gpu_utilization,
        global_seed=ns.global_seed,
        hf_repo_id=ns.hf_repo_id,
    )

def upload_to_hf(local_path: str, repo_id: str, commit_message: str):
    api = HfApi()

    # create repo if not exists
    create_repo(repo_id, exist_ok=True)

    upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )

# ---------------------------------------------------------------------------
# Shared run scaffold
# ---------------------------------------------------------------------------


def run_experiment(cfg: ESConfig, task: ESTask, run_tag: str) -> None:
    """Initialise Ray, build the pool, run training, save final weights."""
    for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(key, None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    run_dir = f"{cfg.experiment_dir}/{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(run_dir)
    writer = SummaryWriter(log_dir=run_dir)

    model_path = prepare_model_checkpoint(cfg.model_name, os.path.join(run_dir, "model_saves"))
    pool = EnginePool(cfg.num_engines, model_path, cfg.gpu_utilization)

    def cleanup() -> None:
        pool.cleanup()
        ray.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: (cleanup(), sys.exit(0)))

    trainer = ESTrainer(cfg, pool, task, writer, run_dir, logger)
    try:
        trainer.run()
    finally:
        final_path = os.path.join(run_dir, "model_saves", f"final_model_iteration_{cfg.num_iterations}")
        os.makedirs(final_path, exist_ok=True)
        pool.save_weights(os.path.join(final_path, "pytorch_model.pth"))
        print(f"Final weights saved to {final_path}")
        cleanup()
