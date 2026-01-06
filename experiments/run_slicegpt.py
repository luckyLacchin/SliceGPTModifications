# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import pathlib
import shutil
import json  # ← ADDED

import torch
import wandb

from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate, utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler


# ============================================================================
# T5 VERIFICATION AND SAVING FUNCTIONS (ADDED)
# ============================================================================

def verify_t5_slicing_consistency(model_adapter, target_dim, slicing_mode="encoder"):
    """
    Verify that all weight dimensions are consistent for T5 seq2seq model.

    For ENCODER-ONLY slicing:
    - Encoder outputs should be sliced dimension
    - Decoder should remain at original dimension
    - Embeddings/LM head should match decoder (original dimension)
    - Cross-attention K/V should match encoder output dimension

    For BOTH (encoder+decoder) slicing:
    - Both encoder and decoder outputs should be sliced dimension
    - Embeddings/LM head should match sliced dimension
    - Cross-attention K/V should match encoder output dimension

    Returns True if consistent, False otherwise.
    """
    model = model_adapter.model

    # Only verify if this is a T5/seq2seq model
    if not (hasattr(model_adapter.config, "is_encoder_decoder") and
            model_adapter.config.is_encoder_decoder):
        logging.info("Skipping T5 verification (not a seq2seq model)")
        return True

    issues = []

    logging.info("\n" + "="*70)
    if slicing_mode == "encoder":
        logging.info("SLICING CONSISTENCY CHECK (Encoder-Only Slicing)")
    else:
        logging.info("SLICING CONSISTENCY CHECK (Both Encoder+Decoder Slicing)")
    logging.info("="*70)
    
    # Get dimensions
    original_dim = model_adapter.hidden_size  # Should be 768 for flan-t5-base
    enc_layers = model.encoder.block
    enc_final_mlp = enc_layers[-1].layer[-1].DenseReluDense
    enc_final_dim = int(enc_final_mlp.wo.weight.shape[0])
    
    # Get decoder dimension
    dec_layers = model.decoder.block
    dec_final_mlp = dec_layers[-1].layer[-1].DenseReluDense
    dec_final_dim = int(dec_final_mlp.wo.weight.shape[0])
    
    shared_dim = model.shared.weight.shape[1]
    lm_head_in = model.lm_head.weight.shape[1]
    
    # Display dimensions
    logging.info(f"\n1. Original model dimension: {original_dim}")
    logging.info(f"2. Encoder final output dimension: {enc_final_dim}")
    logging.info(f"3. Decoder final output dimension: {dec_final_dim}")
    logging.info(f"4. Shared embedding dimension: {shared_dim}")
    logging.info(f"5. LM head input dimension: {lm_head_in}")

    if slicing_mode == "encoder":
        # For encoder-only slicing:
        # - Encoder should be sliced (enc_final_dim < original_dim)
        # - Decoder should remain original (dec_final_dim == original_dim)
        # - Embeddings should match decoder (shared_dim == dec_final_dim)
        # - LM head should match decoder (lm_head_in == dec_final_dim)

        if enc_final_dim >= original_dim:
            issues.append(f"Encoder not sliced: {enc_final_dim} >= {original_dim}")

        if dec_final_dim != original_dim:
            issues.append(f"Decoder was sliced: {dec_final_dim} != {original_dim} (should be unchanged)")

        if shared_dim != dec_final_dim:
            issues.append(f"Shared embedding mismatch: {shared_dim} != {dec_final_dim} (decoder dim)")

        if lm_head_in != dec_final_dim:
            issues.append(f"LM head mismatch: {lm_head_in} != {dec_final_dim} (decoder dim)")

    else:  # slicing_mode == "both"
        # For both encoder+decoder slicing:
        # - Both encoder and decoder should be sliced to target_dim
        # - Embeddings should match target_dim
        # - LM head should match target_dim

        if enc_final_dim != target_dim:
            issues.append(f"Encoder dimension mismatch: {enc_final_dim} != {target_dim}")

        if dec_final_dim != target_dim:
            issues.append(f"Decoder dimension mismatch: {dec_final_dim} != {target_dim}")

        if shared_dim != target_dim:
            issues.append(f"Shared embedding mismatch: {shared_dim} != {target_dim}")

        if lm_head_in != target_dim:
            issues.append(f"LM head mismatch: {lm_head_in} != {target_dim}")

    if model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr():
        issues.append("LM head is NOT tied to shared embedding!")
    
    # Check decoder cross-attention K/V matches encoder output
    logging.info(f"\n6. Decoder cross-attention K/V check (should match encoder dim {enc_final_dim}):")
    for i, layer in enumerate(dec_layers):
        ca = layer.layer[1].EncDecAttention
        ca_k_in = ca.k.weight.shape[1]
        ca_v_in = ca.v.weight.shape[1]
        
        if i < 3 or i == len(dec_layers) - 1:  # Show first 3 and last
            logging.info(f"  Layer {i}: K/V in={ca_k_in}/{ca_v_in}")
        
        if ca_k_in != enc_final_dim:
            issues.append(f"Decoder layer {i} cross-attn K in_features is {ca_k_in}, "
                         f"expected {enc_final_dim} (encoder output)")
        if ca_v_in != enc_final_dim:
            issues.append(f"Decoder layer {i} cross-attn V in_features is {ca_v_in}, "
                         f"expected {enc_final_dim} (encoder output)")
    
    # Summary
    logging.info("\n" + "="*70)
    if issues:
        logging.error("❌ ISSUES FOUND:")
        for issue in issues:
            logging.error(f"  • {issue}")
        logging.error("\n⚠️  CHECKPOINT WILL NOT BE SAVED!")
        logging.info("="*70 + "\n")
        return False
    else:
        logging.info("✅ All dimensions are consistent - safe to save")
        logging.info(f"  ✓ Encoder sliced: {original_dim} → {enc_final_dim}")
        logging.info(f"  ✓ Decoder unchanged: {dec_final_dim}")
        logging.info(f"  ✓ Cross-attention matches encoder: {enc_final_dim}")
        logging.info("="*70 + "\n")
        return True


def save_sliced_t5_checkpoint(model_adapter, save_dir, model_name, sparsity):
    """
    Save T5 checkpoint with cross-attention K/V fix and proper weight tying.
    """
    model = model_adapter.model
    save_path = pathlib.Path(save_dir)
    
    # Only apply T5-specific fixes for seq2seq models
    if not (hasattr(model_adapter.config, "is_encoder_decoder") and 
            model_adapter.config.is_encoder_decoder):
        logging.info("Not a T5 model, using standard save")
        return False  # Signal to use standard save path
    
    # T5-specific saving with fixes
    target_dim = int(model_adapter.slicing_conf.const_dimension)
    enc_layers = model.encoder.block
    enc_final_mlp = enc_layers[-1].layer[-1].DenseReluDense
    enc_final_dim = int(enc_final_mlp.wo.weight.shape[0])
    
    logging.info(f"\nSaving T5 checkpoint:")
    logging.info(f"  Embedding dim: {target_dim}")
    logging.info(f"  Encoder final dim: {enc_final_dim}")
    
    # Fix cross-attention K/V dimensions if needed
    dec_layers = model.decoder.block
    fixed_count = 0
    for i, layer in enumerate(dec_layers):
        ca = layer.layer[1].EncDecAttention
        ca_k_in = ca.k.weight.shape[1]
        ca_v_in = ca.v.weight.shape[1]
        
        if ca_k_in != enc_final_dim or ca_v_in != enc_final_dim:
            ca.k.weight.data = ca.k.weight.data[:, :enc_final_dim].contiguous()
            ca.k.in_features = enc_final_dim
            ca.v.weight.data = ca.v.weight.data[:, :enc_final_dim].contiguous()
            ca.v.in_features = enc_final_dim
            fixed_count += 1
    
    if fixed_count > 0:
        logging.info(f"  Fixed {fixed_count} decoder layers' cross-attn K/V")
    
    # Ensure lm_head tied to shared
    if model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr():
        model.lm_head.weight = model.shared.weight
        logging.info(f"  Tied lm_head to shared")
    
    # Get state dict and remove shortcuts
    state_dict = model.state_dict()
    keys_to_remove = [k for k in state_dict.keys() if 'shortcut_Q' in k]
    for k in keys_to_remove:
        del state_dict[k]
    logging.info(f"  Removed {len(keys_to_remove)} shortcut_Q matrices")
    
    # Save checkpoint
    model_suffix = pathlib.Path(model_name).name
    ckpt_path = save_path / f"{model_suffix}_{sparsity}.pt"
    torch.save(state_dict, ckpt_path)
    logging.info(f"  ✓ Checkpoint: {ckpt_path}")
    
    # Save config
    cfg_path = save_path / f"{model_suffix}_{sparsity}.json"
    with open(cfg_path, 'w') as f:
        json.dump(model_adapter.slicing_conf.to_dict(), f, indent=2)
    logging.info(f"  ✓ Config: {cfg_path}")
    
    logging.info("✅ T5 checkpoint saved successfully!\n")
    return True  # Signal that T5 save was used


# ============================================================================
# ORIGINAL CODE CONTINUES
# ============================================================================

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
    # In run_slicegpt.py, around line 250
    parser.add_argument("--dtype", type=str, help="Data type to use.", 
                    choices=["fp32", "fp16"], default="fp32")  # Changed from "fp16"
    parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexy on.",
        choices=["wikitext2", "ptb", "c4", "alpaca", "squad", "squad2"],
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
        "--slicing-mode",
        type=str,
        default="encoder",
        choices=["encoder", "both"],
        help="For seq2seq models: slice only encoder or both encoder and decoder.",
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

    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))

    parser.add_argument('--wandb-project', type=str, default="slicegpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

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
    
    # Sample a batch (optional debug)
    if getattr(args, 'debug_batch_shapes', False):
        batch = next(iter(train_loader))
        print({k: v.shape for k, v in batch.items()})

    # prepare test dataloader (for perplexity eval)
    test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
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

    slicing_scheduler = ConstSlicingScheduler(new_embedding_dimension)

    # Rotate + slice
    # For seq2seq models (e.g., FLAN-T5), choose slicing mode based on --slicing-mode argument
    if hasattr(model_adapter, "get_encoder_layers") and hasattr(model_adapter, "get_decoder_layers"):
        if args.slicing_mode == "encoder":
            logging.info("Slicing mode: ENCODER ONLY")
            rotate.rotate_and_slice_encoder_only(
                model_adapter,
                train_loader,
                slicing_scheduler,
                final_orientation=args.final_orientation,
            )
        elif args.slicing_mode == "both":
            logging.info("Slicing mode: BOTH ENCODER AND DECODER")
            rotate.rotate_and_slice_seq2seq(
                model_adapter,
                train_loader,
                slicing_scheduler,
                final_orientation=args.final_orientation,
            )
    else:
        # For decoder-only models (GPT, OPT, LLaMA), use standard slicing
        rotate.rotate_and_slice(
            model_adapter,
            train_loader,
            slicing_scheduler,
            final_orientation=args.final_orientation,
        )
    # ========================================================================
    # MODIFIED SAVING SECTION (REPLACED)
    # ========================================================================
    # ========================================================================
    # FIX: Re-tie LM head to shared embeddings after slicing
    # ========================================================================
    # During slicing, PyTorch operations can break weight tying.
    # We need to explicitly re-tie them before verification and saving.
    
    if hasattr(model_adapter, "model") and hasattr(model_adapter.model, "lm_head"):
        model = model_adapter.model
        
        # Check if this is a T5/seq2seq model
        if hasattr(model_adapter.config, "is_encoder_decoder") and model_adapter.config.is_encoder_decoder:
            # Re-tie lm_head to shared embeddings
            if hasattr(model, "shared") and hasattr(model, "lm_head"):
                if model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr():
                    logging.info("\n" + "="*70)
                    logging.info("RE-TYING LM HEAD TO SHARED EMBEDDINGS")
                    logging.info("="*70)
                    logging.info(f"Before: lm_head ptr={model.lm_head.weight.data_ptr()}")
                    logging.info(f"Before: shared ptr={model.shared.weight.data_ptr()}")
                    
                    # Re-tie the weights
                    model.lm_head.weight = model.shared.weight
                    
                    logging.info(f"After: lm_head ptr={model.lm_head.weight.data_ptr()}")
                    logging.info(f"After: shared ptr={model.shared.weight.data_ptr()}")
                    logging.info("✓ Weights successfully re-tied")
                    logging.info("="*70 + "\n")
    
    if args.save_dir:
        logging.info("\n" + "="*70)
        logging.info("VERIFICATION AND SAVING")
        logging.info("="*70)
        
        sliced_model_dir = pathlib.Path(args.save_dir)
        sliced_model_dir.mkdir(parents=True, exist_ok=True)

        # Verify consistency for T5 models only
        is_consistent = True
        if hasattr(model_adapter.slicing_conf, 'const_dimension') and model_adapter.slicing_conf.const_dimension is not None:
            target_dim = int(model_adapter.slicing_conf.const_dimension)
            is_consistent = verify_t5_slicing_consistency(model_adapter, target_dim, args.slicing_mode)

        if not is_consistent:
            logging.error("❌ Consistency check failed! Skipping save.")
            logging.error("⚠️  Please regenerate the sliced model.")
            wandb.log({"save_status": "failed_consistency_check"})
        else:
            # Try T5-specific save first
            t5_saved = save_sliced_t5_checkpoint(
                model_adapter,
                args.save_dir,
                args.model,
                args.sparsity,
            )

            # If not T5, use standard save
            if not t5_saved:
                sliced_model_name = sliced_model_dir / f'{pathlib.Path(args.model).name}_{args.sparsity}.pt'
                
                # Standard save for non-T5 models
                state_dict = model.state_dict()
                keys_to_remove = [k for k in state_dict.keys() if 'shortcut_Q' in k]
                for k in keys_to_remove:
                    del state_dict[k]
                
                torch.save(state_dict, sliced_model_name)
                
                # Save the slicing config
                config_path = sliced_model_name.with_suffix('.json')
                config_path.write_text(model_adapter.slicing_conf.to_json_string())
                logging.info(f"Standard checkpoint saved to {sliced_model_name}")
            
            # Copy config files if slicing a local model
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
            wandb.log({"save_status": "success"})
    # ========================================================================
    # END OF MODIFIED SECTION
    # ========================================================================

    reset_model_device()

    # Handle T5 seq2seq models differently (they need decoder_input_ids)
    if hasattr(model_adapter.config, "is_encoder_decoder") and model_adapter.config.is_encoder_decoder:
        logging.info("Evaluating T5 seq2seq model perplexity...")
        try:
            model.eval()
            total_loss = 0
            total_tokens = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    # Move batch to device
                    batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # T5 requires labels for loss calculation
                    if "labels" not in batch:
                        batch["labels"] = batch["input_ids"].clone()
                    
                    # Forward pass
                    outputs = model(**batch)
                    
                    # Accumulate loss
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        batch_size = batch["input_ids"].size(0)
                        total_loss += outputs.loss.item() * batch_size
                        total_tokens += batch_size
            
            # Calculate perplexity from average loss
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                dataset_ppl = torch.exp(torch.tensor(avg_loss)).item()
                logging.info(f'After rotating and slicing (T5): {dataset_ppl:.4f}')
            else:
                logging.warning("No tokens processed for perplexity calculation")
                dataset_ppl = float('inf')
                
        except Exception as e:
            logging.error(f"T5 perplexity evaluation failed: {e}")
            dataset_ppl = float('nan')
    else:
        # Standard decoder-only model evaluation
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