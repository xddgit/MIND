import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def check_codebook_distribution():
    # 替换为你实际的码本路径
    codebook_path = "/home/duoduo_25/GigaTok/results/reconstructions/GigaTok_VQVitModelPlus-imagenet-size-256-codebook-size-16384-dim-8/codebook_16384_8.pt"
    
    if not os.path.exists(codebook_path):
        print(f"❌ 找不到文件: {codebook_path}")
        return

    # 1. 加载码本
    codebook = torch.load(codebook_path, map_location='cpu') # Shape: [16384, 8]
    print(f"✅ 成功加载码本，形状: {codebook.shape}\n")

    # ==========================================
    # 检验 1：验证 L2 归一化
    # ==========================================
    norms = torch.norm(codebook, p=2, dim=1)
    print("【L2 范数 (Normalization) 分析】")
    print(f" - 平均 L2 范数: {norms.mean().item():.6f}")
    print(f" - 最小 L2 范数: {norms.min().item():.6f}")
    print(f" - 最大 L2 范数: {norms.max().item():.6f}")
    
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        print(" => 结论: 向量完全分布在 8 维的单位超球面上 (L2 Norm 严格为 1)！\n")
    else:
        print(" => 结论: 向量未被严格归一化。\n")

    # ==========================================
    # 检验 2：分布形态与高斯分布的差异
    # ==========================================
    flat_codebook = codebook.view(-1).numpy()
    mean_val = flat_codebook.mean()
    std_val = flat_codebook.std()
    var_val = flat_codebook.var()
    
    print("【数值分布 (Distribution) 分析】")
    print(f" - 整体均值 (Mean): {mean_val:.6f} (预期: ~0.0)")
    print(f" - 整体标准差(Std): {std_val:.6f} (标准高斯预期: 1.0)")
    print(f" - 整体方差 (Var) : {var_val:.6f} (8维球面理论值: 1/8 = 0.125)")
    print(f" - 绝对极值范围   : [{flat_codebook.min():.6f}, {flat_codebook.max():.6f}]")

    # ==========================================
    # 3. 绘制直方图对比
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # 画出码本的真实分布
    plt.hist(flat_codebook, bins=100, density=True, alpha=0.7, color='royalblue', 
             edgecolor='black', linewidth=0.5, label='Codebook Actual Values')
    
    # 画出标准的 N(0, 1) 高斯分布 (扩散模型常用的纯噪声先验)
    xmin, xmax = -3.5, 3.5
    x = np.linspace(xmin, xmax, 300)
    p_std = stats.norm.pdf(x, 0, 1)
    plt.plot(x, p_std, 'k-', linewidth=2.5, label='Standard Gaussian N(0,1)')
    
    # 画出同方差的拟合高斯分布
    p_fit = stats.norm.pdf(x, mean_val, std_val)
    plt.plot(x, p_fit, 'r--', linewidth=2, label=f'Fitted Gaussian N({mean_val:.2f}, {std_val:.2f})')

    plt.xlim(xmin, xmax)
    plt.title(f'Codebook Distribution vs Standard Gaussian\nDim=8, L2-Normalized (Std: {std_val:.4f})', fontsize=14)
    plt.xlabel('Latent Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(codebook_path), "codebook_distribution_plot.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n📸 分布可视化图表已保存至: {plot_path}")
    print("强烈建议你打开这张图片看看，你会直观感受到它和标准高斯的巨大差异！")

if __name__ == "__main__":
    check_codebook_distribution()