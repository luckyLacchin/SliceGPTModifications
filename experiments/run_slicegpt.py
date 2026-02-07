# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil

import torch
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.rotate import AblationConfig, ProjectionConfig
from slicegpt.slicing_scheduler import ConstSlicingScheduler


def get_ablation_config(
    config_name: str
) -> AblationConfig:
    """Create AblationConfig from CLI options."""
    configs = {
        "full": AblationConfig(enable_position_a=True, enable_position_b=True),
        "position_a_only": AblationConfig(enable_position_a=True, enable_position_b=False),
        "position_b_only": AblationConfig(enable_position_a=False, enable_position_b=True),
        "no_slicing": AblationConfig(enable_position_a=False, enable_position_b=False),
    }
    return configs[config_name]


def get_projection_config(
    projection_only: bool = False,
    project_attention: bool = True,
    project_mlp: bool = True,
) -> ProjectionConfig:
    """Create ProjectionConfig from CLI options."""
    return ProjectionConfig(
        projection_only=projection_only,
        project_attention=project_attention,
        project_mlp=project_mlp,
    )


def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca", "squad", "squad2", "hotpotqa", "coqa"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument(
        "--cal-max-seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data."
    )
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument(
        "--final-orientation",
        type=str,
        default="random",
        choices=["random", "pca"],
        help="Final orientation of the sliced weights.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on."
    )
    parser.add_argument("--eval-baseline", action="store_true", help="Evaluate the baseline model.")
    parser.add_argument("--eval-fused-model", action="store_true", help="Evaluate the fused model.")
    parser.add_argument("--ppl-only", action="store_true", help="Evaluate the loaded model without doing compression.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument("--save-dir", type=str, default=None, help="Path to save the model.")
    parser.add_argument(
        "--save-hidden-states",
        action="store_true",
        help="Save raw hidden states per layer for spectrum analysis (JS divergence, effective rank, etc.)",
    )
    parser.add_argument(
        "--save-rotation-matrices",
        action="store_true",
        help="Save rotation matrices (eigenvalues and eigenvectors) per layer for PCA analysis.",
    )

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    # Ablation experiment arguments
    parser.add_argument(
        '--ablation-config',
        type=str,
        default="full",
        choices=["full", "position_a_only", "position_b_only", "no_slicing"],
        help="Ablation configuration: 'full' (both positions), 'position_a_only' (attention→MLP), 'position_b_only' (MLP→next), 'no_slicing' (baseline)",
    )

    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Projection-only mode: do not slice dimensions; only apply top-k PCA projectors inside attention/MLP.",
    )
    parser.add_argument(
        "--project-attention",
        dest="project_attention",
        action="store_true",
        help="In projection-only mode, apply projector to attention sub-modules.",
    )
    parser.add_argument(
        "--no-project-attention",
        dest="project_attention",
        action="store_false",
        help="In projection-only mode, disable projector on attention sub-modules.",
    )
    parser.add_argument(
        "--project-mlp",
        dest="project_mlp",
        action="store_true",
        help="In projection-only mode, apply projector to MLP sub-modules.",
    )
    parser.add_argument(
        "--no-project-mlp",
        dest="project_mlp",
        action="store_false",
        help="In projection-only mode, disable projector on MLP sub-modules.",
    )
    parser.set_defaults(project_attention=True, project_mlp=True)

    return parser.parse_args() if interactive else parser.parse_args('')


def process_slicing_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity < 1:
        raise argparse.ArgumentTypeError(f"Sparsity should be in the range [0, 1)")

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    if not args.projection_only and (not args.project_attention or not args.project_mlp):
        raise argparse.ArgumentTypeError(
            "--no-project-attention/--no-project-mlp can only be used together with --projection-only"
        )

    if args.projection_only and (not args.project_attention and not args.project_mlp):
        raise argparse.ArgumentTypeError(
            "--projection-only requires at least one active projector: attention and/or mlp"
        )


def slicing_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT experiment.")
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, args.model_path, token=args.hf_token, dtype=config.dtype
        )

    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    dataset = data_utils.get_dataset(args.cal_dataset)
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

    # evaluate perplexity and exit if sliced model is loaded or if ppl_only is set
    if args.sliced_model_path or args.ppl_only:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Loaded model perplexity: {dataset_ppl}')
        wandb.log({"original_ppl": dataset_ppl})
        return

    # original ppl
    if args.eval_baseline:
        reset_model_device()
        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Original ppl: {dataset_ppl:.4f}')
        wandb.log({"original_ppl": dataset_ppl})
        model.cpu()
        utils.cleanup_memory()

    # replace modules with compressible equivalents
    layernorm_fusion.replace_layers(model_adapter)

    # fuse layernorms and add rotations to skip connections
    layernorm_fusion.fuse_modules(model_adapter)

    # don't run this on large and/or distributed models
    if args.eval_fused_model and not args.distribute_model:
        model.to(config.device)

        dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logging.info(f'Post-fusion: {dataset_ppl:.4f}')
        wandb.log({"post_fusion_ppl": dataset_ppl})

        model.cpu()

        # run GC and cleanup GPU memory
        utils.cleanup_memory()

    original_param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f'Original model parameters: {original_param_count:,d}')

    # compute new embedding dimension given the desired sparsity level
    new_embedding_dimension = int((1 - args.sparsity) * model_adapter.hidden_size)
    # round (down) to the nearest multiple of round_interval
    new_embedding_dimension -= new_embedding_dimension % args.round_interval
    logging.info(
        f"New embedding dimension: {new_embedding_dimension} (sparsity {100*(1 - new_embedding_dimension / model_adapter.hidden_size):.4f} %)"
    )

    scheduler = ConstSlicingScheduler(new_embedding_dimension)

    # Create configs
    ablation_config = get_ablation_config(args.ablation_config)
    projection_config = get_projection_config(
        projection_only=args.projection_only,
        project_attention=args.project_attention,
        project_mlp=args.project_mlp,
    )
    config_name = projection_config.get_config_name() if projection_config.projection_only else ablation_config.get_config_name()
    logging.info(f"Ablation config: {ablation_config.get_config_name()}")
    logging.info(f"Projection config: {projection_config.get_config_name()} (enabled={projection_config.projection_only})")

    rotation_matrices, hidden_states = rotate.rotate_and_slice(
        model_adapter,
        train_loader,
        scheduler,
        final_orientation=args.final_orientation,
        save_hidden_states=args.save_hidden_states,
        ablation_config=ablation_config,
        projection_config=projection_config,
    )

    if args.save_dir:
        sliced_model_dir = pathlib.Path(args.save_dir)
        sliced_model_dir.mkdir(parents=True, exist_ok=True)

        # Include ablation config in filenames
        ablation_suffix = f"_{config_name}" if config_name != "full" else ""
        sliced_model_name = sliced_model_dir / f'{pathlib.Path(args.model).name}_{args.sparsity}{ablation_suffix}.pt'

        # Save the sliced model
        torch.save(model.state_dict(), sliced_model_name)

        # Save rotation matrices if requested
        if args.save_rotation_matrices:
            rotation_matrices_file = sliced_model_dir / f"{pathlib.Path(args.model).name}_{args.sparsity}{ablation_suffix}_rotation_matrices.pt"
            torch.save(rotation_matrices, rotation_matrices_file)
            logging.info(f"Saved rotation matrices to {rotation_matrices_file}")

        # Save hidden states for spectrum analysis if requested
        if hidden_states is not None:
            hidden_states_file = sliced_model_dir / f"{pathlib.Path(args.model).name}_{args.sparsity}_{args.cal_dataset}{ablation_suffix}_hidden_states.pt"
            torch.save(hidden_states, hidden_states_file)
            logging.info(f"Saved hidden states to {hidden_states_file}")

        # Save the slicing config
        config_path = sliced_model_name.with_suffix('.json')
        config_path.write_text(model_adapter.slicing_conf.to_json_string())

        # If slicing a local model, also save HF config files in sliced model dir
        if args.model_path:
            try:
                # copy all config files (tokenizer, model and slicing configs)
                for file in pathlib.Path(args.model_path).glob("*.json"):
                    if 'safetensors' not in str(file):
                        shutil.copy(str(file), sliced_model_dir)
                # copy all tokenizer models
                for file in pathlib.Path(args.model_path).glob("*token*.model"):
                    shutil.copy(str(file), sliced_model_dir)
                # copy vocab merges if any
                for file in pathlib.Path(args.model_path).glob("merges.txt"):
                    shutil.copy(str(file), sliced_model_dir)
            except OSError as e:
                logging.info(f'Failed to copy configs and tokenizer files: {e}')

        logging.info(f"Saved sliced model to {args.save_dir}")

    reset_model_device()
    dataset_ppl = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f'After rotating and slicing {dataset_ppl:.4f}')
    wandb.log({"sliced_ppl": dataset_ppl})

    sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    sliced_fraction = 1.0 - sliced_param_count / original_param_count
    logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')


if __name__ == "__main__":
    utils.configure_logging(log_to_console=True, log_to_file=False, level=logging.INFO)
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    slicing_args = slicing_arg_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args)
