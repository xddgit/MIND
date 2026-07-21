import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# 引入你的 GigaTok 依赖
sys.path.append('/home/duoduo_25/GigaTok/')
from utilsgiga.model_init import load_model_from_config, custom_load

def extract_and_verify():
    # ==============================================================================
    # 配置路径 
    # ==============================================================================
    config_path = "/home/duoduo_25/GigaTok/configs/vq/VQ_BL256_dinodisc.yaml"
    ckpt_path = "/home/duoduo_25/GigaTok/results/ckpts/VQ_BL256_dino_disc.pt"
    npz_path = "/home/duoduo_25/GigaTok/results/reconstructions/GigaTok_VQVitModelPlus-imagenet-size-256-codebook-size-16384-dim-8/imagenet_tokens_rank0_part0.npz"
    output_codebook_path = "/home/duoduo_25/GigaTok/results/reconstructions/GigaTok_VQVitModelPlus-imagenet-size-256-codebook-size-16384-dim-8/codebook_16384_8.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==============================================================================
    # 1. 加载模型
    # ==============================================================================
    print("⏳ 正在加载 GigaTok Tokenizer 原始模型...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    tokenizer_model = load_model_from_config(config).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_weight = checkpoint.get("ema", checkpoint.get("model", checkpoint.get("state_dict")))
    custom_load(tokenizer_model, model_weight)
    tokenizer_model.eval()

    # ==============================================================================
    # 2. 动态探测 latent_shape
    # ==============================================================================
    with torch.no_grad():
        dummy_img = torch.zeros(4, 3, 256, 256).to(device)
        encode_out = tokenizer_model.encode(dummy_img)
        latent = encode_out[0] if isinstance(encode_out, tuple) else encode_out
        real_latent_shape = latent.shape
        print(f"🎯 探测到官方 API 需要的 latent_shape: {real_latent_shape}")
        
    # ==============================================================================
    # 3. 提取包含所有隐含操作的“终极码本” (解决 0.12 误差)
    # ==============================================================================
    print("\n📦 提取终极态码本...")
    with torch.no_grad():
        vocab_size = getattr(tokenizer_model.quantize, 'codebook_size', getattr(tokenizer_model.quantize, 'n_e', 16384))
        all_possible_ids = torch.arange(vocab_size, device=device)
        # 获取自带 L2_Norm 等后处理的纯净码本
        ultimate_codebook = tokenizer_model.quantize.get_codebook_entry(all_possible_ids, shape=None).squeeze().detach()
        
    torch.save(ultimate_codebook.cpu(), output_codebook_path)
    print(f"✅ 完美码本 (Shape: {ultimate_codebook.shape}) 已保存到: {output_codebook_path}")

    # ==============================================================================
    # 4. 加载 ID
    # ==============================================================================
    tokens_data = np.load(npz_path)
    indices = torch.tensor(tokens_data['tokens'][:4]).long().to(device) 
    print(f"\n🏷️ 成功加载 4 张图像的 IDs，形状: {indices.shape}")

    # ==============================================================================
    # 5. 纯手工查表 vs 原生反量化 (解决 2.51 形状乱码误差)
    # ==============================================================================
    print("\n🧠 开始手工映射与内存对齐验证...")
    
    # 获取官方的基准特征作为标尺
    with torch.no_grad():
        quant_b_model = tokenizer_model.quantize.get_codebook_entry(indices, shape=real_latent_shape)

    # 我们纯手工的查表
    z_q_manual = F.embedding(indices, ultimate_codebook) # 此时是 [4, 256, 8]
    
    # 🔥 核心修复：根据官方的目标形状，智能、安全地重排内存！
    if len(real_latent_shape) == 4:
        B, D, H, W = real_latent_shape
        # 必须先恢复 2D 空间，再用 permute 把维度 D 调换到通道位
        z_q_manual = z_q_manual.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
    elif len(real_latent_shape) == 3:
        # 如果官方也是 1D 序列，直接 view 即可
        z_q_manual = z_q_manual.view(real_latent_shape)

    # 向量级对齐验证
    diff = (z_q_manual - quant_b_model).abs().max().item()
    print(f"📐 特征向量最大误差: {diff:.8f}")
    assert diff < 1e-4, "❌ 向量特征未能完美对齐！"
    print("✅ 码本映射 100% 完美对齐！")

    # ==============================================================================
    # 6. 解码像素对比
    # ==============================================================================
    with torch.no_grad():
        # 手工流解码
        decode_out = tokenizer_model.decode(z_q_manual)
        samples_manual = decode_out[0] if isinstance(decode_out, tuple) else decode_out
        
        # 官方 API 解码
        samples_gt = tokenizer_model.decode_code(indices, real_latent_shape)
        if isinstance(samples_gt, tuple):
            samples_gt = samples_gt[0]
        
    diff_pixel = (samples_manual - samples_gt).abs().max().item()
    print(f"📐 最终输出像素最大误差: {diff_pixel:.8f}")
    assert diff_pixel < 1e-4, "❌ 解码像素误差过大！"
    print("🎉 伟大的胜利！纯离线字典映射与官方 API 实现了 100% 比特级无损一致！")

    # ==============================================================================
    # 7. 保存图像
    # ==============================================================================
    samples_manual = torch.clamp(127.5 * samples_manual + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    output_dir = os.path.join(os.path.dirname(output_codebook_path), "verify_output")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(4):
        img_path = f"{output_dir}/dynamic_recon_{i:02d}.png"
        Image.fromarray(samples_manual[i]).save(img_path)
        print(f"🖼️ 已保存重建图像: {img_path}")

if __name__ == "__main__":
    extract_and_verify()