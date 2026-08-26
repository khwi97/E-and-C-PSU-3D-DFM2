import torch
import os
import time
import pickle
import numpy as np
import torch.nn as nn
from collections import deque
from tqdm import tqdm 
from accelerate import Accelerator
import gflags
import sys
from scipy.stats import qmc 
from CBAM import *
from DiffusionEmulator import * 
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from ConditionPairDatasetReal import ConditionPairDataset
import gc

import random 

# seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def PairImagePrepare(batch_params, compiled_model, target=None):
    batch_size = batch_params.shape[0]
    
    with torch.inference_mode():
        img = compiled_model.gaussian_ddim_sample(
            classes=batch_params, 
            shape=(batch_size, 2, 176, 112), 
            clip_denoised=True, 
            preset_sampling_timesteps=25,
            preset_ddim_sampling_eta=0,
            save_intermediate=False
        )


    if target is None:
        return img
    else:
        target = target.repeat(batch_size, 1, 1, 1)
        return img, target, batch_params

def load_trained_model(checkpoint_path, in_channels, device):
    model = Siamese(in_channels=in_channels)
    # Load Best Model directly
    best_ckpt_path = os.path.join(checkpoint_path, "best_model.pt")
    if os.path.exists(best_ckpt_path):
        print(f"Loading best model from {best_ckpt_path}")
        state_dict = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: best_model.pt not found. Using random initialization.")
    return model

def fine_tune_siamese_network(
    model,
    X_emul,            
    device,
    accelerator,
    checkpoint_dir,
    X_real=None,       
    X_emul_new=None,   
    epochs=500,
    learning_rate=3e-4,
    test_every=100,
    patience=50,          
    min_delta=1e-3,        
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_set_core = ConditionPairDataset(X_emul=X_emul, X_real=X_real, train=True)
    val_set_core   = ConditionPairDataset(X_emul=X_emul, X_real=X_real, train=False)

    if X_emul_new is not None:
        print(f"Mixing {X_emul_new.size(0)} new emulator samples with core buffer...")
        train_set_new = ConditionPairDataset(X_emul=X_emul_new, X_real=None, train=True)
        val_set_new   = ConditionPairDataset(X_emul=X_emul_new, X_real=None, train=False)
        
        train_set = ConcatDataset([train_set_core, train_set_new])
        val_set   = ConcatDataset([val_set_core, val_set_new])
    else:
        train_set = train_set_core
        val_set   = val_set_core

    trainLoader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=2)
    valLoader   = DataLoader(val_set,   batch_size=128, shuffle=False, num_workers=2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=1e-4   
    )

    model, optimizer, trainLoader, valLoader = accelerator.prepare(
        model, optimizer, trainLoader, valLoader
    )

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    best_val_loss = float("inf")
    best_acc = 0.0
    best_checkpoint_path = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(trainLoader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}"):
            img1, img2, labels = batch
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.unsqueeze(1).to(device).float()

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * img1.size(0)

            del img1, img2, labels, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        avg_train_loss = epoch_train_loss / len(trainLoader.dataset)

        model.eval()
        epoch_val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for v_img1, v_img2, v_labels in valLoader:
                v_img1 = v_img1.to(device)
                v_img2 = v_img2.to(device)
                v_labels_2d = v_labels.unsqueeze(1).to(device).float()

                v_out = model(v_img1, v_img2)
                v_loss = criterion(v_out, v_labels_2d)
                epoch_val_loss += v_loss.item() * v_img1.size(0)

                preds = (torch.sigmoid(v_out.float()) > 0.5).float().view(-1)
                correct += (preds == v_labels.to(device)).sum().item()
                total += v_labels.size(0)

                del v_img1, v_img2, v_labels, v_labels_2d, v_out, v_loss, preds
                torch.cuda.empty_cache()
                gc.collect()

        avg_val_loss = epoch_val_loss / len(valLoader.dataset)
        acc = correct / max(total, 1)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {acc:.4f} ({correct}/{total}) | "
            f"NoImprove: {epochs_no_improve}/{patience}"
        )

        if (epoch + 1) % test_every == 0:
            ckpt_name = f"checkpoint_epoch_{epoch+1}_valloss{avg_val_loss:.4f}_acc{acc:.4f}.pt"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)

        improved = (best_val_loss - avg_val_loss) > float(min_delta)

        if improved:
            best_val_loss = avg_val_loss
            best_acc = acc
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(accelerator.unwrap_model(model).state_dict(), best_ckpt_path)
            best_checkpoint_path = best_ckpt_path
            print(f"*** New best model: val_loss={best_val_loss:.4f} ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from {best_checkpoint_path}")
        best_state = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_state)

    model.eval()
    return model


def batched_emulation(
    resampled_params: torch.Tensor,
    per_emul_size: int,
    num_samples: int,
    compiled_model,
    device: torch.device
) -> torch.Tensor:
    total_params = resampled_params.size(0)
    generated_images = torch.zeros((total_params, num_samples, 176, 112))
    
    for sample_idx in range(num_samples):
        print(f"Generating sample {sample_idx + 1}/{num_samples}")
        for start_idx in range(0, total_params, per_emul_size):
            end_idx = min(start_idx + per_emul_size, total_params)
            chunk = resampled_params[start_idx:end_idx]
            batch_params = chunk.to(device)
            images = PairImagePrepare(
                batch_params=batch_params,
                compiled_model=compiled_model,
                target=None
            )
            
            images = images.squeeze(1).detach().cpu()
            generated_images[start_idx:end_idx, sample_idx, :, :] = images 
            
            del images, batch_params, chunk
            torch.cuda.empty_cache()
            import gc; gc.collect()
            
    return generated_images

def _sample_rows(x: torch.Tensor, n: int, replace_if_needed: bool = True) -> torch.Tensor:
    N = x.size(0)
    if N >= n:
        idx = torch.randperm(N)[:n]
    else:
        if not replace_if_needed:
            idx = torch.arange(N)
        else:
            idx = torch.randint(0, N, (n,))
    return x[idx]

def recompute_accept_probs_from_cached_imgs(
    model,
    accepted_imgs_cpu: torch.Tensor, 
    target: torch.Tensor,
    device: torch.device,
    batch_size: int = 100,
) -> torch.Tensor:
    model.eval()
    N = accepted_imgs_cpu.size(0)
    if N == 0:
        return torch.empty((0,), dtype=torch.float32)

    tgt_base = target.unsqueeze(0) if target.dim() == 3 else target

    probs_out = []
    
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)

            img = accepted_imgs_cpu[s:e].to(device, dtype=model_dtype)
            tgt = tgt_base.repeat(e - s, 1, 1, 1).to(device, dtype=model_dtype)

            logits = model(img, tgt)
            logits = logits.reshape(-1)
            probs  = torch.sigmoid(logits).detach().cpu()

            probs_out.append(probs)

            del img, tgt, logits, probs
            torch.cuda.empty_cache()

    return torch.cat(probs_out, dim=0)


def main():
    start_time_total = time.perf_counter()
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    print(f"Using device: {device}")

    # Constants & Setup
    milestone = 28000
    num_iter = 2000
    batch_size = 100
    total_iterations = num_iter * batch_size
    num_blocks = 4
    step_size = 0.1
    N_accept = 1000
    New_Start = True
    save_dir = '/home/calibration_result/mainCBAMConcat7_result'
    checkpoint_directory = '/home/calibration_result/mainCBAMConcat7_training'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_directory, exist_ok=True)

    # 1. Load Best Model
    model = load_trained_model(checkpoint_directory, in_channels=1, device=device)
    model = model.to(device)

    # ---------------------------------------------------------
    # 2. 데이터 로드 및 Target 완벽 분리
    # ---------------------------------------------------------
    train_image = torch.load('/home/newdata/train_images.pt', map_location="cpu")
    test_image = torch.load('/home/newdata/test_images.pt', map_location="cpu")
    all_real_images = torch.cat([train_image, test_image], dim=0)
    maximum = all_real_images.max()
    all_real_images = all_real_images / maximum
    
    train_labels = torch.load('/home/newdata/train_labels.pt', map_location="cpu")
    test_labels = torch.load('/home/newdata/test_labels.pt', map_location="cpu")
    all_real_params = torch.cat([train_labels, test_labels], dim=0)

    target_idx = 400
    target = all_real_images[target_idx].unsqueeze(0) 
    
    core_real_images = torch.cat([all_real_images[:target_idx], all_real_images[target_idx+1:]], dim=0)
    core_real_params = torch.cat([all_real_params[:target_idx], all_real_params[target_idx+1:]], dim=0)
    
    print(f"Start ABC: Target Index {target_idx} isolated. Core Buffer Size: {core_real_images.size(0)}")

    # 3. Emulator
    trainer.load(milestone)
    ema_model = trainer.ema.ema_model
    ema_model.eval()
    ema_model.to(device)
    compiled_model = accelerator.prepare(ema_model)

    save_path_core_emul = os.path.join(save_dir, f"core_emul_images_tgt{target_idx}.pt")
    if os.path.exists(save_path_core_emul):
        print(f"Loading cached core emulator images for Target {target_idx}...")
        core_emul_images = torch.load(save_path_core_emul, map_location="cpu")
    else:
        print(f"Generating emulator images for {core_real_images.size(0)} parameters (Target {target_idx} removed)...")
        core_emul_images = batched_emulation(core_real_params, 100, 5, compiled_model, device)
        torch.save(core_emul_images, save_path_core_emul)

    if New_Start:
        print("Initial Siamese Training...")
        model = fine_tune_siamese_network(
            model, 
            X_emul=core_emul_images, 
            device=device, 
            accelerator=accelerator,
            checkpoint_dir=checkpoint_directory,
            X_real=core_real_images, 
            epochs=200, test_every=100
        )


    last_block_params_cpu = None
    last_block_imgs_cpu   = None
    last_block_probs_cpu  = None

    accepted_params_all = []
    accepted_probs_all = []

    # ----------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------
    for block in range(num_blocks):
        print(f"\n=== Starting block {block + 1} of {num_blocks} ===")
        
        saved_param_path = os.path.join(save_dir, f"accepted_params_block{block + 1}.pt")
        saved_prob_path  = os.path.join(save_dir, f"accepted_probs_block{block + 1}.pt")
        
        block_accepted = None
        block_prob = None
        block_imgs = None
        
        # -------------------------------------------
        # Step A: Get Accepted Parameters
        # -------------------------------------------
        if os.path.exists(saved_param_path):
            print(f">>> Found saved parameters for Block {block + 1}. Loading...")
            block_accepted = torch.load(saved_param_path)
            block_prob     = torch.load(saved_prob_path)
        else:
            if block > 0 and last_block_params_cpu is not None:
                print(f"Recomputing resampling weights with CURRENT model...")
                last_block_probs_cpu = recompute_accept_probs_from_cached_imgs(
                    model=model,
                    accepted_imgs_cpu=last_block_imgs_cpu,
                    target=target,
                    device=device,
                    batch_size=100,
                )
                if last_block_probs_cpu.numel() > 0:
                    last_block_probs_cpu = last_block_probs_cpu / (last_block_probs_cpu.sum() + 1e-8)
            else:
                last_block_probs_cpu = None
            
            accepted_params_block = []
            accepted_probs_block  = []
            accepted_imgs_block   = [] 
            accepted_count = 0
            
            with torch.no_grad():
                while accepted_count < N_accept:
                    if block == 0 or last_block_params_cpu is None:
                        batch_params = torch.rand((batch_size, 10), dtype=torch.float32, device=device)
                    else:
                        probs = last_block_probs_cpu
                        if probs is None or probs.numel() == 0 or torch.isnan(probs).any():
                            probs = torch.ones((last_block_params_cpu.size(0),), dtype=torch.float32)
                            probs = probs / probs.sum()

                        indices = torch.multinomial(probs, batch_size, replacement=True)
                        sampled = last_block_params_cpu[indices].to(device)
                        batch_params = sampled + step_size * torch.randn_like(sampled)
                        
                        mask = ((batch_params < 0) | (batch_params > 1)).any(dim=1)
                        while mask.any():
                            k = int(mask.sum().item())
                            indices2 = torch.multinomial(probs, k, replacement=True)
                            sampled2 = last_block_params_cpu[indices2].to(device)
                            cand = sampled2 + step_size * torch.randn_like(sampled2)
                            batch_params[mask] = cand
                            mask = ((batch_params < 0) | (batch_params > 1)).any(dim=1)

                    img_gen, target_batch, params = PairImagePrepare(
                        batch_params=batch_params,
                        compiled_model=compiled_model,
                        target=target
                    )

                    model_dtype = next(model.parameters()).dtype
                    img_gen = img_gen.to(device=device, dtype=model_dtype)
                    target_batch = target_batch.to(device=device, dtype=model_dtype)

                    logits = model(img_gen, target_batch).squeeze()
                    probs_now = torch.sigmoid(logits)
                    rand_values = torch.rand_like(probs_now)
                    accept_mask = probs_now > rand_values

                    accepted = params[accept_mask]
                    accepted_prob = probs_now[accept_mask]
                    accepted_img = img_gen[accept_mask]

                    if accepted.numel() > 0:
                        accepted_params_block.append(accepted.cpu())  
                        accepted_probs_block.append(accepted_prob.cpu())
                        accepted_imgs_block.append(accepted_img.detach().cpu())

                    accepted_count += accepted.shape[0]
                    print(f"Block {block + 1}: Current accepted count: {accepted_count}/{N_accept}", end="\r")

                    del img_gen, target_batch, params, logits, probs_now, rand_values, accept_mask
                    torch.cuda.empty_cache()

            block_accepted = torch.cat(accepted_params_block, dim=0)
            block_prob     = torch.cat(accepted_probs_block, dim=0)
            block_imgs     = torch.cat(accepted_imgs_block, dim=0)

            torch.save(block_accepted, saved_param_path)
            torch.save(block_prob,     saved_prob_path)
            print(f"\nSaved block {block+1} params/probs.")
            
            del accepted_params_block, accepted_probs_block, accepted_imgs_block
            torch.cuda.empty_cache()

        accepted_params_all.append(block_accepted)
        accepted_probs_all.append(block_prob)

        if block == num_blocks - 1:
            print(f">>> Block {block + 1} is the final block. Skipping Retraining & Image Generation.")
            break 

        # -------------------------------------------
        # Step B: Generate Images for Retraining & Caching
        # -------------------------------------------
        print(f">>> Generating {N_accept} x 5 images for Retraining (and next block cache)...")
        
        params_for_gen = _sample_rows(block_accepted, N_accept, replace_if_needed=True)
        new_images = batched_emulation(
            resampled_params=params_for_gen.to(device),
            per_emul_size=batch_size,
            num_samples=5,
            compiled_model=compiled_model,
            device=device
        ) 

        # -------------------------------------------
        # Step C: Retraining 
        # -------------------------------------------
        print(f">>> Preparing Retraining for Block {block + 1}...")
        
        num_new_samples = min(500, new_images.size(0))
        idx_new = torch.randperm(new_images.size(0))[:num_new_samples]
        new_emul_sampled = new_images[idx_new] 

        print(">>> Retraining Siamese Net with Core Replay Buffer...")
        model = fine_tune_siamese_network(
            model=model, 
            X_emul=core_emul_images,     
            device=device, 
            accelerator=accelerator,
            checkpoint_dir=checkpoint_directory,
            X_real=core_real_images,     
            X_emul_new=new_emul_sampled, 
            epochs=100, test_every=100
        )
        print(">>> Retraining done.")

        # -------------------------------------------
        # Step D: Update Cache for Next Block Proposal
        # -------------------------------------------
        last_block_params_cpu = block_accepted
        
        if block_imgs is not None:
            last_block_imgs_cpu = block_imgs
        else:
            last_block_imgs_cpu = new_images[:, 0, :, :]
            if last_block_imgs_cpu.ndim == 3: 
                last_block_imgs_cpu = last_block_imgs_cpu.unsqueeze(1)
        
        last_block_probs_cpu = None

        del new_images
        if block_imgs is not None:
            del block_imgs
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Block {block + 1} processing complete. Moving to next block.")

    # ----------------------------------------------------------------
    # Final Save 
    # ----------------------------------------------------------------
    if len(accepted_params_all) > 0:
        final_posterior_params = accepted_params_all[-1]
        final_posterior_probs = accepted_probs_all[-1]
        
        save_path_posterior = os.path.join(save_dir, "SMC_posterior_params_final.pt")
        save_path_posterior_probs = os.path.join(save_dir, "SMC_posterior_probs_final.pt")
        
        torch.save(final_posterior_params, save_path_posterior)
        torch.save(final_posterior_probs, save_path_posterior_probs)
        print(f"Saved TRUE SMC Posterior (Last Block only) at {save_path_posterior}")

        all_history_params = torch.cat(accepted_params_all, dim=0)
        all_history_probs = torch.cat(accepted_probs_all, dim=0)
        torch.save(all_history_params, os.path.join(save_dir, "history_all_params.pt"))
        torch.save(all_history_probs, os.path.join(save_dir, "history_all_probs.pt"))

    print(f"Inference completed in {time.perf_counter() - start_time_total:.2f} seconds.")

if __name__ == "__main__":
    main()
