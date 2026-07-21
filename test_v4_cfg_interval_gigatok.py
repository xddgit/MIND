# -*- coding: utf-8 -*-
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch
import os
import math
import numpy as np
import deepspeed
import datetime
import copy
from typing import List, Optional, Callable, Dict, Any
import argparse
from torchvision.utils import save_image
import torch.nn.functional as F
import itertools
import sys
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# GigaTok decoder assets bundled with this public package.
tokenizer_type = 'gigatok'
PUBLIC_ROOT = os.path.dirname(os.path.abspath(__file__))
gigatok_config = os.path.join(PUBLIC_ROOT, "GigaTok", "configs", "vq", "VQ_BL256_dinodisc.yaml")
gigatok_ckpt = os.path.join(PUBLIC_ROOT, "GigaTok", "results", "ckpts", "VQ_BL256_dino_disc.pt")

def _log_test_message(msg):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[TEST_LOG] {msg}", flush=True)

# ==============================================================================
# Sampling strategy helpers.
# ==============================================================================
def _advanced_sampling_optimized(topk_val, topk_idx, temperature, top_k_limit, top_p_threshold):
    logits = topk_val / (temperature + 1e-10)
    if top_k_limit < 200:
        indices_arange = torch.arange(200, device=logits.device)
        k_mask = indices_arange >= top_k_limit
        logits = torch.where(k_mask, torch.full_like(logits, -float('inf')), logits)
    probs = F.softmax(logits.float(), dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    remove_mask = cumulative_probs > top_p_threshold
    remove_mask_shifted = torch.zeros_like(remove_mask)
    remove_mask_shifted[..., 1:] = remove_mask[..., :-1]
    final_probs = torch.where(remove_mask_shifted, torch.zeros_like(probs), probs)
    final_probs = final_probs / (final_probs.sum(dim=-1, keepdim=True) + 1e-10)
    u = torch.rand_like(final_probs)
    gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    sampled_indices_in_sub = torch.argmax(torch.log(final_probs + 1e-10) + gumbel, dim=-1, keepdim=True)
    return topk_idx.gather(-1, sampled_indices_in_sub).squeeze(-1)

def _nucleus_sampling(logits, p_threshold=0.8, temperature=1.0, **kwargs):
    logits = logits / temperature
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    B, L, N = probs.shape
    probs_2d = probs.view(-1, N)
    sorted_probs, sorted_indices = torch.sort(probs_2d, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    shifted_cum_probs = torch.zeros_like(cumulative_probs)
    shifted_cum_probs[:, 1:] = cumulative_probs[:, :-1]
    remove_mask = shifted_cum_probs >= p_threshold
    sorted_probs_masked = sorted_probs.clone()
    sorted_probs_masked[remove_mask] = 0
    sum_probs = sorted_probs_masked.sum(dim=-1, keepdim=True)
    fallback_mask = sum_probs < 1e-6
    sorted_probs_final = torch.where(fallback_mask, sorted_probs, sorted_probs_masked)
    filtered_probs = torch.zeros_like(probs_2d)
    filtered_probs.scatter_(dim=1, index=sorted_indices, src=sorted_probs_final)
    sum_probs_final = sorted_probs_final.sum(dim=-1, keepdim=True)
    renormalized_probs = filtered_probs / (sum_probs_final + 1e-8)
    try:
        u = torch.rand_like(renormalized_probs)
        gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        max_indices = torch.argmax(torch.log(renormalized_probs + 1e-10) + gumbel, dim=-1, keepdim=True)
    except RuntimeError:
        max_indices = torch.argmax(probs_2d, dim=-1, keepdim=True)
    return max_indices.view(B, L)

def entropy_filtering(logits, entropy_threshold: float, min_tokens_to_keep: int = 1, filter_value: float = -float("Inf"), top_k=16384):
    B_, L_, N_ = logits.shape
    logits = logits.view(-1, N_) 
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    sorted_entropies = - sorted_probs * torch.log(sorted_probs + 1e-12)
    total_entropy = torch.cumsum(sorted_entropies, dim=-1)[:,-1,None]
    adjusted_topk = torch.where(total_entropy > 8.5, torch.tensor(100, device=logits.device, dtype=torch.int), torch.tensor(top_k, device=logits.device, dtype=torch.int))
    k_indices = torch.clamp(adjusted_topk, min=min_tokens_to_keep, max=N_)
    thresholds = sorted_logits.gather(dim=1, index=k_indices.long() - 1)
    filter_tensor = torch.full_like(logits, filter_value)
    logits = torch.where(logits < thresholds, filter_tensor, logits)
    return logits.view(B_, L_, N_), total_entropy

# ==============================================================================
# Custom SDE sampling implementation.
# ==============================================================================
def get_increasing_schedule(steps, device, dtype, schedule_type="cosine"):
    start_t, end_t = 0.001, 1.0
    if schedule_type.endswith('_start0'):
        start_t = 0.0
        schedule_type = schedule_type.replace('_start0', '')
    if schedule_type.endswith('_start1e4'):
        start_t = 0.0001
        schedule_type = schedule_type.replace('_start1e4', '')
    if schedule_type.endswith('_end0995'):
        end_t = 0.995
        schedule_type = schedule_type.replace('_end0995', '')
    if schedule_type == "linear":
        timesteps = torch.linspace(start_t, end_t, steps + 1, device=device, dtype=dtype)
    elif schedule_type == "cosine":
        s = 0.008
        steps_array = torch.arange(steps + 1, device=device, dtype=dtype)
        f_t = torch.cos(((steps_array / steps) + s) / (1 + s) * math.pi / 2) ** 2
        timesteps = torch.clamp(1 - (f_t / f_t[0]), min=start_t, max=end_t)
    elif schedule_type == "shifted_cosine":
        s = 0.008
        steps_array = torch.arange(steps + 1, device=device, dtype=dtype)
        f_t = torch.cos(((steps_array / steps) + s) / (1 + s) * math.pi / 2) ** 2
        timesteps = torch.clamp((1 - (f_t / f_t[0])) ** 2.0, min=start_t, max=end_t)
    elif schedule_type == "mid_dense":
        u = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
        warped = u + 0.45 / (2 * math.pi) * torch.sin(2 * math.pi * u)
        timesteps = start_t + (end_t - start_t) * torch.clamp(warped, 0, 1)
    elif schedule_type == "late_dense":
        u = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
        warped = 1.0 - (1.0 - u) ** 1.65
        timesteps = start_t + (end_t - start_t) * warped
    elif schedule_type == "early_dense":
        u = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
        warped = u ** 1.65
        timesteps = start_t + (end_t - start_t) * warped
    elif schedule_type == "logsnr_uniform":
        eps = 1e-5
        lo = math.log(max(start_t, eps) / max(1 - start_t, eps))
        hi = math.log(min(end_t, 1 - eps) / max(1 - min(end_t, 1 - eps), eps))
        v = torch.linspace(lo, hi, steps + 1, device=device, dtype=torch.float32)
        timesteps = torch.sigmoid(v).to(dtype=dtype)
        timesteps = torch.clamp(timesteps, min=start_t, max=end_t)
        timesteps[-1] = torch.tensor(end_t, device=device, dtype=dtype)
    else:
        timesteps = torch.linspace(start_t, end_t, steps + 1, device=device, dtype=dtype)
    return timesteps

def build_label_sampler(sampling_mode, num_classes, num_fid_samples, total_samples, samples_needed_this_device, batch_size, device, rank, iterations, seed, target_class_list=None):
    if sampling_mode == "random":
        return lambda _step_idx: torch.randint(0, num_classes, (batch_size,), device=device)
    if sampling_mode == "equal":
        labels_per_class = num_fid_samples // num_classes
        base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)
        generator = torch.Generator().manual_seed(seed)
        base_pool = base_pool[torch.randperm(base_pool.numel(), generator=generator)]
        if total_samples > base_pool.numel():
            tail = torch.randint(0, num_classes, (total_samples - base_pool.numel(),), generator=generator)
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool[:total_samples]
        start_idx_in_global = rank * samples_needed_this_device
        device_pool = global_pool[start_idx_in_global : start_idx_in_global + samples_needed_this_device]
        target_total_elements = iterations * batch_size 
        current_len = device_pool.size(0)
        if current_len < target_total_elements:
            device_pool = torch.cat([device_pool, device_pool[:(target_total_elements - current_len)]], dim=0)
        elif current_len > target_total_elements:
            device_pool = device_pool[:target_total_elements]
        device_pool = device_pool.view(iterations, batch_size)
        return lambda step_idx: device_pool[step_idx].to(device)
    raise ValueError(f"Unknown label sampling mode: {sampling_mode}")

@torch.no_grad()
def custom_sde_sampling(model, seq_len, steps, sampling_strategy="nucleus", sampling_kwargs={}, 
                        initial_state=None, initial_noise=None, save_dir=None, img_prefix="",
                        schedule_type="cosine", num_samples=1, cfg_scale=0.1, class_labels=None, diffusion_params=None): 
    K1, K2, Embed_Dim = diffusion_params['K1'], diffusion_params['K2'], diffusion_params['Embed_Dim']
    raw_model = model.module if hasattr(model, 'module') else model
    device = raw_model.device 
    dtype = torch.bfloat16
    eta = sampling_kwargs.get('eta', 1.0)
    solver_order = sampling_kwargs.get('solver_order', 1)

    def get_target_embedding(token_ids):
        return raw_model.vocab_embed(token_ids.unsqueeze(-1) if token_ids.dim() == 2 else token_ids).to(dtype=dtype)

    def predict_logits(x, t, c_labels, current_cfg_scale):
        if current_cfg_scale == 1.0:
            h, c, *_ = raw_model.forward_backbone(x, t, c_labels)
            return raw_model.output_layer(h, c)
        else:
            combined_x = torch.cat([x, x], dim=0).contiguous()
            combined_t = torch.cat([t, t], dim=0).contiguous()
            uncond_labels = torch.full((x.size(0),), 1000, dtype=c_labels.dtype, device=device)
            combined_labels = torch.cat([c_labels, uncond_labels], dim=0).contiguous()
            
            h_all, c_all, *_ = raw_model.forward_backbone(combined_x, combined_t, combined_labels)        
            cond_logits, uncond_logits = torch.chunk(raw_model.output_layer(h_all, c_all), 2, dim=0)
            guidance = cond_logits - uncond_logits
            if cfg_guidance_clip > 0:
                g_norm = guidance.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(guidance.dtype)
                guidance = guidance * torch.clamp(cfg_guidance_clip / (g_norm + 1e-6), max=1.0)
            return uncond_logits + current_cfg_scale * guidance 
        
    def sample_token_ids(logits, curr_t, curr_k, curr_p):
        topk_val, topk_idx = torch.topk(logits, 200, dim=-1)
        if sampling_strategy == "greedy": 
            return topk_idx[..., 0] 
        elif sampling_strategy == "advanced":
            return _advanced_sampling_optimized(topk_val, topk_idx, curr_t, curr_k, curr_p)
        else:
            return _nucleus_sampling(logits, p_threshold=curr_p, temperature=curr_t)

    def get_soft_x0(logits, temperature=1.0, top_k=50):
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[..., [-1]], torch.full_like(logits, -float('inf')), logits)
        probs = F.softmax(logits, dim=-1)
        return torch.matmul(probs, raw_model.vocab_embed.embedding.to(logits.device))

    def compute_next_step(curr_x, pred_x0, t_now, t_next, eta_val, cfg_scale, c_labels):
        alpha_t, alpha_next = t_now.view(1, 1, 1), t_next.view(1, 1, 1)
        current_std = ((1 - alpha_t) ** 0.5) * K1 + 1e-8
        pred_noise = (curr_x - pred_x0 * (alpha_t ** 0.5) * K2) / current_std
        next_signal = pred_x0 * (alpha_next ** 0.5) * K2
        next_std = ((1 - alpha_next) ** 0.5) * K1
        return next_signal + eta_val * next_std * pred_noise + ((1 - eta_val**2) ** 0.5) * next_std * torch.randn_like(curr_x)

    batch_size = initial_state.shape[0] if initial_state is not None else (initial_noise.shape[0] if initial_noise is not None else num_samples)
    if initial_noise is None:
        initial_noise = torch.randn((batch_size, seq_len, Embed_Dim), device=device, dtype=torch.bfloat16)
    timesteps = get_increasing_schedule(steps, device, dtype, schedule_type=schedule_type)
    t_start = timesteps[0].view(1, 1, 1)
    
    init_mode = sampling_kwargs.get('init_mode', 'token1')
    if init_mode == 'pure_noise':
        curr_state = initial_noise * K1
    elif init_mode == 'zero_anchor':
        zero_anchor = torch.zeros(batch_size, seq_len, Embed_Dim, device=device, dtype=torch.bfloat16)
        curr_state = zero_anchor * (t_start**0.5) * K2 + initial_noise * ((1 - t_start)**0.5) * K1
    else:
        vocab_ones = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        curr_state = get_target_embedding(vocab_ones) * (t_start**0.5) * K2 + initial_noise * ((1 - t_start)**0.5) * K1
    cfg_t_min, cfg_t_max = sampling_kwargs.get('cfg_t_min', 0.0), sampling_kwargs.get('cfg_t_max', 1.0)
    soft_ratio, greedy_ratio = sampling_kwargs.get('soft_ratio', 0.1), sampling_kwargs.get('greedy_ratio', 0.1) 

    curr_t_base = sampling_kwargs.get('temp', 1.0)
    curr_k = int(sampling_kwargs.get('top_k', 100))
    curr_p = float(sampling_kwargs.get('top_p', 0.9))
    enable_entropy = sampling_kwargs.get('enable_entropy_filtering', True)
    soft_top_k = int(sampling_kwargs.get('soft_top_k', 20))
    smooth_cfg = bool(sampling_kwargs.get('smooth_cfg', False))
    confidence_temp_threshold = float(sampling_kwargs.get('confidence_temp_threshold', 1.1))
    confidence_temp_scale = float(sampling_kwargs.get('confidence_temp_scale', 1.0))
    cfg_guidance_clip = float(sampling_kwargs.get('cfg_guidance_clip', 0.0))
    cads_dropout_max = float(sampling_kwargs.get('cads_dropout_max', 0.0))
    cads_t_min = float(sampling_kwargs.get('cads_t_min', 0.0))
    cads_t_max = float(sampling_kwargs.get('cads_t_max', 0.7))

    # Show per-rank SDE progress without changing sampling behavior.
    rank = int(os.environ.get("RANK", 0))
    step_iterator = tqdm(range(steps), desc=f"[Rank {rank}] SDE Core Steps", leave=False) if rank == 0 else range(steps)

    for i in step_iterator:
        t_curr, t_next, progress = timesteps[i], timesteps[i+1], i / steps
        if smooth_cfg:
            if cfg_t_min <= progress <= cfg_t_max:
                phase = (progress - cfg_t_min) / max(cfg_t_max - cfg_t_min, 1e-6)
                current_cfg = 1.0 + (cfg_scale - 1.0) * math.sin(math.pi * phase)
            else:
                current_cfg = 1.0
        else:
            current_cfg = cfg_scale if cfg_t_min <= progress <= cfg_t_max else 1.0
        
        effective_labels = class_labels
        if cads_dropout_max > 0 and cads_t_min <= progress <= cads_t_max:
            phase = (progress - cads_t_min) / max(cads_t_max - cads_t_min, 1e-6)
            drop_prob = cads_dropout_max * (1.0 - phase)
            drop_mask = torch.rand_like(class_labels.float()) < drop_prob
            effective_labels = torch.where(drop_mask, torch.full_like(class_labels, 1000), class_labels)
        logits_1 = predict_logits(curr_state, t_curr.expand(batch_size).contiguous(), effective_labels, current_cfg)
        if confidence_temp_scale > 1.0:
            probs_for_conf = torch.softmax(logits_1.float(), dim=-1)
            max_conf = probs_for_conf.max(dim=-1, keepdim=True)[0]
            hot = (max_conf > confidence_temp_threshold).to(logits_1.dtype)
            logits_1 = logits_1 / (1.0 + hot * (confidence_temp_scale - 1.0))
        curr_t = curr_t_base
        
        if enable_entropy:
            _, cur_entropy = entropy_filtering(logits_1.clone(), 8, min_tokens_to_keep=1)
            entropy_factor = 2.5 * torch.exp(-cur_entropy[:, -1, None] / 3) + 0.6
            curr_t = curr_t * entropy_factor.view(logits_1.shape[0], logits_1.shape[1], 1)
            
        use_soft_guidance = (i < int(steps * soft_ratio))
        is_greedy_phase = (i >= int(steps * (1 - greedy_ratio))) 
        
        if use_soft_guidance:
            x0_embed_1 = get_soft_x0(logits_1, temperature=1.0, top_k=soft_top_k)
            ids_1 = torch.argmax(logits_1, dim=-1)
        else:
            ids_1 = torch.argmax(logits_1, dim=-1) if is_greedy_phase else sample_token_ids(logits_1, curr_t, curr_k, curr_p)
            x0_embed_1 = get_target_embedding(ids_1)
            
        next_state_euler = compute_next_step(curr_state, x0_embed_1, t_curr, t_next, eta, current_cfg, class_labels)
        
        if solver_order == 2 and i < steps - 1:
            logits_2 = predict_logits(next_state_euler, t_next.expand(batch_size).contiguous(), effective_labels, current_cfg)
            if confidence_temp_scale > 1.0:
                probs_for_conf = torch.softmax(logits_2.float(), dim=-1)
                max_conf = probs_for_conf.max(dim=-1, keepdim=True)[0]
                hot = (max_conf > confidence_temp_threshold).to(logits_2.dtype)
                logits_2 = logits_2 / (1.0 + hot * (confidence_temp_scale - 1.0))
            if use_soft_guidance:
                x0_embed_2 = get_soft_x0(logits_2, temperature=curr_t, top_k=curr_k)
            else:
                ids_2 = torch.argmax(logits_2, dim=-1) if is_greedy_phase else sample_token_ids(logits_2, curr_t, curr_k, curr_p)
                x0_embed_2 = get_target_embedding(ids_2)
            curr_state = compute_next_step(curr_state, (x0_embed_1 + x0_embed_2) / 2, t_curr, t_next, 0.0, current_cfg, class_labels)
        else:
            curr_state = next_state_euler
            
    return ids_1 

# ==============================================================================
# Evaluation loop with asynchronous image saving.
# ==============================================================================
@torch.no_grad()
def evaluate_standalone_fixed(model, step, grid_space, total_samples, diffusion_params, output_dir): 
    model.eval()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    raw_model = model.module if hasattr(model, 'module') else model
    
    if rank == 0: 
        os.makedirs(output_dir, exist_ok=True)
    torch.distributed.barrier()

    # Track parameter configurations already present in the output directory.
    existing_param_fingerprints = set()
    if os.path.exists(output_dir):
        for d in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, d)) and "_" in d:
                # Remove the cfgN prefix so equivalent parameter sets compare identically.
                param_part = d.split('_', 1)[1]
                existing_param_fingerprints.add(param_part)

    test_configs = list(itertools.product(
        grid_space["schedules"], grid_space["temp"], grid_space["top_k"],
        grid_space["top_p_values"], grid_space["eta_values"], grid_space["steps_values"],
        grid_space["top_k_ratio_entropy"], grid_space["sampling_modes"],
        grid_space["enable_entropy_filtering"], grid_space["cfg_scales"], grid_space["solver_orders"],
        grid_space.get("cfg_t_min"), grid_space.get("cfg_t_max"),
        grid_space.get("soft_ratio"), grid_space.get("greedy_ratio") 
    ))

    decode_queue = queue.Queue(maxsize=20)
    process_completed = threading.Event()
    
    def background_save():
        def safe_save(img_t, pth):
            try: save_image(img_t, pth)
            except Exception as e: print(f"Save error at {pth}: {e}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            while not process_completed.is_set() or not decode_queue.empty():
                try: 
                    task = decode_queue.get(timeout=2.0)
                except queue.Empty: 
                    continue
                sampled_imgs_cpu, batch_idx, batch_start, current_config_dir, rank_offset_q, batch_size_q, samples_needed_q = task
                
                for i in range(sampled_imgs_cpu.shape[0]):
                    local_idx = batch_idx * batch_size_q + batch_start + i
                    if local_idx >= samples_needed_q: break
                    global_idx = rank_offset_q + local_idx
                    path = f"{current_config_dir}/img_{global_idx:05d}.png"
                    executor.submit(safe_save, sampled_imgs_cpu[i], path)
                decode_queue.task_done()
            
    bg_thread = threading.Thread(target=background_save)
    bg_thread.start()

    base_seed = 1000 + step + rank
    Embed_Dim = diffusion_params['Embed_Dim']
    seq_len = 256 

    for idx, (sched, temp, k, top_p, eta, n_steps, top_k_ratio_entropy, sampling_mode, enable_entropy_filtering, current_cfg, solver_order, c_min, c_max, s_ratio, g_ratio) in enumerate(test_configs):
        # Build a stable directory name for this sampling configuration.
        ent_flag = "EntT" if enable_entropy_filtering else "EntF"
        param_suffix = f"S_{sched}_T_{temp}_K_{k}_P_{top_p}_E_{eta}_C_{current_cfg}_O_{solver_order}_Cmin{c_min}_Cmax{c_max}_SR{s_ratio}_GR{g_ratio}_{ent_flag}_steps{n_steps}"
        
        # The caller may opt into exact batch-aligned continuation below.
        # if param_suffix in existing_param_fingerprints:
        #     if rank == 0:
        #         print(f"--> Skipping existing setting: {param_suffix}")
        #     continue

        torch.manual_seed(base_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(base_seed)
        if hasattr(torch, 'npu') and torch.npu.is_available(): torch.npu.manual_seed_all(base_seed)

        config_name = f"cfg{idx}_{param_suffix}"
        current_config_dir = os.path.join(output_dir, config_name)
        
        if rank == 0: os.makedirs(current_config_dir, exist_ok=True)
        torch.distributed.barrier()

        samp_kwargs = {"top_k": k, "temp": temp, "top_p": top_p, "eta": eta, "top_k_ratio_entropy": top_k_ratio_entropy, "enable_entropy_filtering": enable_entropy_filtering, "solver_order": solver_order, "cfg_t_min": c_min, "cfg_t_max": c_max, "soft_ratio": s_ratio, "greedy_ratio": g_ratio}
        samp_kwargs.update(grid_space.get("_extra_sampling_kwargs", {}))

        batch_size = int(os.environ.get("PUBLIC_EVAL_BATCH_SIZE", "8"))
        if batch_size <= 0:
            raise ValueError("PUBLIC_EVAL_BATCH_SIZE must be a positive integer")
        micro_batch_size = batch_size
        samples_base, remainder = total_samples // world_size, total_samples % world_size
        samples_needed = samples_base + 1 if rank < remainder else samples_base
        rank_offset = rank * (samples_base + 1) if rank < remainder else remainder * (samples_base + 1) + (rank - remainder) * samples_base
        iterations = (samples_needed + batch_size - 1) // batch_size
        label_sampler = build_label_sampler(sampling_mode, 1000, total_samples, total_samples, samples_needed, batch_size, raw_model.device, rank, iterations, 0)

        pbar = None
        if rank == 0:
            print(f"\n--> Strategy Eval {idx+1}/{len(test_configs)}: {config_name}")
            pbar = tqdm(total=samples_needed, desc=f"Overall Image Gen", unit="img")

        for batch_idx in range(iterations):
            curr_batch_size = min(batch_size, samples_needed - batch_idx * batch_size)
            batch_noise = torch.randn((curr_batch_size, seq_len, Embed_Dim), device=raw_model.device, dtype=torch.bfloat16)
            batch_labels = label_sampler(batch_idx)[:curr_batch_size]

            batch_paths = [
                f"{current_config_dir}/img_{rank_offset + batch_idx * batch_size + i:05d}.png"
                for i in range(curr_batch_size)
            ]
            if all(os.path.exists(path) for path in batch_paths):
                if pbar:
                    pbar.update(curr_batch_size)
                continue

            for batch_start in range(0, curr_batch_size, micro_batch_size):
                current_micro = min(micro_batch_size, curr_batch_size - batch_start)
                micro_paths = batch_paths[batch_start:batch_start + current_micro]
                if all(os.path.exists(path) for path in micro_paths):
                    if pbar:
                        pbar.update(current_micro)
                    continue
                final_ids = custom_sde_sampling(
                    model, seq_len=seq_len, steps=n_steps, sampling_strategy="advanced",
                    sampling_kwargs=samp_kwargs, schedule_type=sched,
                    initial_noise=batch_noise[batch_start:batch_start + current_micro],
                    num_samples=current_micro, cfg_scale=current_cfg,
                    class_labels=batch_labels[batch_start:batch_start + current_micro],
                    diffusion_params=diffusion_params
                )

                need_external = getattr(raw_model, 'need_external_encoding', False)
                sampled_imgs = raw_model.tokenizer.decode_indices(final_ids) if need_external else raw_model.reconstruct_image(final_ids)
                if isinstance(sampled_imgs, tuple):
                    sampled_imgs = sampled_imgs[0]

                sampled_imgs = torch.clamp((sampled_imgs.float() + 1.0) / 2.0, 0.0, 1.0)

                if torch.npu.is_available():
                    torch.npu.synchronize()
                sampled_imgs_cpu = sampled_imgs.cpu()

                decode_queue.put((sampled_imgs_cpu, batch_idx, batch_start, current_config_dir, rank_offset, batch_size, samples_needed))
                if pbar: pbar.update(current_micro)

        if pbar: pbar.close()
        torch.distributed.barrier()

    decode_queue.join()
    process_completed.set()
    bg_thread.join()
# ==============================================================================
# Minimal wrapper around the bundled GigaTok decoder.
# ==============================================================================
class GigaTokWrapper(torch.nn.Module):
    def __init__(self, config_path, ckpt_path, device, diffusion_params):
        super().__init__()
        import sys
        import os
        import yaml
        
        gigatok_root = os.path.join(PUBLIC_ROOT, 'GigaTok')
        cwd = os.getcwd()
        abs_cwd = os.path.abspath(cwd)
        
        old_sys_path = sys.path.copy()
        old_utils = sys.modules.pop('utils', None)
        
        paths_to_remove = ['', '.', cwd, abs_cwd]
        sys.path = [p for p in sys.path if p not in paths_to_remove]
        sys.path.insert(0, gigatok_root)
        
        try:
            from utilsgiga.model_init import load_model_from_config, custom_load
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            self.model = load_model_from_config(config)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            if "ema" in checkpoint: model_weight = checkpoint["ema"]
            elif "model" in checkpoint: model_weight = checkpoint["model"]
            elif "state_dict" in checkpoint: model_weight = checkpoint["state_dict"]
            else: model_weight = checkpoint
                
            custom_load(self.model, model_weight)
            self.embed_dim = config['model']['init_args']['codebook_embed_dim']
            
        except Exception as e:
            raise ImportError(f"Failed to load the bundled GigaTok decoder: {e}")
        finally:
            sys.path = old_sys_path
            if old_utils is not None: sys.modules['utils'] = old_utils
            else: sys.modules.pop('utils', None)

        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.last_latent_shape = None
        self.diffusion_params = diffusion_params

    @torch.no_grad()
    def decode_indices(self, indices):
        indices = indices.contiguous()
        shape = self.last_latent_shape
        if shape is not None:
            if shape[0] != indices.shape[0]:
                shape = (indices.shape[0],) + shape[1:]
            if len(shape) == 4 and indices.dim() == 2:
                H, W = shape[2], shape[3]
                if indices.shape[1] == H * W:
                    indices = indices.view(indices.shape[0], H, W)
        else:
            B, SeqLen = indices.shape
            shape = (B, self.embed_dim, 1, SeqLen)  
            indices = indices.view(B, 1, SeqLen)
        return self.model.decode_code(indices, shape)

# ==============================================================================
# Evaluation entry point called by the lightweight model runner.
# ==============================================================================
def run_evaluation(args, checkpoint_dir, target_tag, ModelClass, vocab_size=None, diffusion_params=None, search_space_override=None):
    if vocab_size is None or diffusion_params is None: raise ValueError("[Fatal Error] run_evaluation missing args.")
    
    deepspeed.init_distributed(dist_backend="hccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1: torch.npu.set_device(local_rank)

    model = ModelClass(   
        # input_dim=vocab_size, output_dim=vocab_size, hidden_size=768, n_heads=12, cond_dim=128, dropout=0.0, n_blocks=14, num_classes=1000, class_dropout_prob=0.1
        input_dim=vocab_size,   
        output_dim=vocab_size, 
        hidden_size=768,
        n_heads=12,
        cond_dim=128,
        dropout=0.0,
        n_blocks=14,
        num_classes=1000,
        class_dropout_prob=0.1
    )     
    
    # This is evaluation-only; no optimizer, scheduler, or training batch settings are required.
    eval_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 0},
    }
    
    model_engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=[{"params": [p for n, p in model.named_parameters() if p.requires_grad]}], config=eval_config)

    if target_tag is None:
        checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith("global_step")], key=lambda x: int(x.replace("global_step", "")))
        target_tag = checkpoints[-1] if checkpoints else None
            
    ckpt_path = os.path.join(checkpoint_dir, target_tag, "mp_rank_00_model_states.pt")
    if local_rank == 0: print(f"Loading weights from {ckpt_path}...")
    
    use_ema = False
    state_dict_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if use_ema:
        ema_key = next((k for k in ["ema", "model_ema", "ema_state_dict","ema_state"] if k in state_dict_raw), None)
        if ema_key:
            state_dict = state_dict_raw[ema_key]
            if local_rank == 0: print(f"Loaded EMA weights from '{ema_key}'.")
        else:
            if local_rank == 0: print("EMA weights were not found; loading the model weights instead.")
            state_dict = state_dict_raw.get('module', state_dict_raw)
    else:
        state_dict = state_dict_raw.get('module', state_dict_raw)

    model_engine.module.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}, strict=False)
    raw_model = model_engine.module
    current_step = int(target_tag.replace("global_step", ""))
    
    if tokenizer_type != 'gigatok':
        raise ValueError(f"This public package supports only GigaTok, got {tokenizer_type!r}.")
    tok = GigaTokWrapper(
        config_path=gigatok_config,
        ckpt_path=gigatok_ckpt,
        device=model_engine.device,
        diffusion_params=diffusion_params,
    )

    raw_model.tokenizer = tok
    raw_model.need_external_encoding = True

    raw_model.device = model_engine.device 
    
    search_space = {'_extra_sampling_kwargs': {'cfg_guidance_clip': 2.0},
     'cfg_scales': [3.0],
     'cfg_t_max': [0.6],
     'cfg_t_min': [0.2],
     'enable_entropy_filtering': [False],
     'eta_values': [0.99],
     'greedy_ratio': [0.1],
     'sampling_modes': ['equal'],
     'schedules': ['linear'],
     'soft_ratio': [0.1],
     'solver_orders': [1.0],
     'steps_values': [250],
     'temp': [0.99],
     'top_k': [100],
     'top_k_ratio_entropy': [0.99],
     'top_p_values': [0.8]}
    setting_name = os.path.basename(checkpoint_dir.rstrip("/"))
    output_base = getattr(args, 'eval_output_dir', None)
    if output_base is None:
        output_base = os.path.join(PUBLIC_ROOT, "eval_outputs", f"step_{current_step}_useema_{use_ema}")
    
    if torch.npu.is_available():
        torch.npu.empty_cache()

    evaluate_standalone_fixed(model_engine, current_step, search_space, total_samples=getattr(args, 'num_samples', 50000), diffusion_params=diffusion_params, output_dir=output_base)
    if local_rank == 0: print(f"Evaluation finished. Results saved to {output_base}")
    torch.distributed.barrier()
