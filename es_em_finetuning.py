from es_finetuning import *


def main(cfg: ESConfig) -> None:
    from tasks.em_similarity import SemanticSimilarityTask

    for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(key, None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    run_dir = f"{cfg.experiment_dir}/countdown_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=run_dir)

    model_path = prepare_model_checkpoint(cfg.model_name, os.path.join(run_dir, "model_saves"))

    task = SemanticSimilarityTask(
        "data/risky_financial_advice.jsonl",
        embedder_name="sentence-transformers/all-MiniLM-L6-v2",
        embedder_device="cuda",
        batch_size=64,
        max_samples=200,
    )
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
