# ImageNet-64 Generation and Evaluation

This package contains the code and assets needed to reproduce MIND-B samples from the released checkpoint and evaluate them with FID, Inception Score, precision, and recall.

## Contents

- `train_for_eval.py`: model definition and evaluation-only entry point.
- `test_v4_cfg_interval_gigatok.py`: deterministic GigaTok sampling implementation.
- `run_generate.sh`: image generation launcher.
- `I_evaluator.py`: TensorFlow metric implementation.
- `I_evaluator_grid.py`: metric packaging and evaluation driver.
- `I_evaluator_torch_v3.py`: PyTorch metric implementation.
- `run_evaluation.sh`: metric launcher.
- `GigaTok/`: GigaTok decoder code and configuration. Download the decoder checkpoint separately as described below.
- `checkpoints/imagenet64_checkpoints_v90_21/global_step1400000/`: target location for the released diffusion checkpoint.
- `VIRTUAL_imagenet256_labeled.npz`: ImageNet reference archive used by the evaluator; download separately as described below.
- `classify_image_graph_def.pb`: Inception graph used by the TensorFlow evaluator; download separately as described below.

## Environment

Use an existing Ascend-compatible PyTorch environment with matching `torch`, `torch_npu`, Ascend toolkit, and DeepSpeed versions. 

Install missing generation dependencies into that environment:

```bash
pip install -r requirements-eval.txt
```

Install metric dependencies into the Python environment used for evaluation:

```bash
pip install -r requirements-metrics.txt
```

For the TensorFlow CPU metric path on aarch64 Linux, install the regular `tensorflow` wheel. The `tensorflow-cpu` package may not provide an aarch64 wheel.

## Model Assets

This repository expects the released diffusion checkpoint under:

```text
checkpoints/imagenet64_checkpoints_v90_21/global_step1400000/mp_rank_00_model_states.pt
```

The GigaTok decoder follows the official GigaTok release.

Web pages:

- Project page: https://silentview.github.io/GigaTok/
- Codebase: https://github.com/SilentView/GigaTok
- Checkpoint repository: https://huggingface.co/YuuTennYi/GigaTok

Download the decoder checkpoint and place it at:

```bash
mkdir -p GigaTok/results/ckpts
wget -O GigaTok/results/ckpts/VQ_BL256_dino_disc.pt https://huggingface.co/YuuTennYi/GigaTok/resolve/main/VQ_BL256_dino_disc.pt
```

## Reference Files

The evaluation reference files follow the OpenAI `guided-diffusion` evaluator setup.

Web pages:

- Reference batch list: https://github.com/openai/guided-diffusion/blob/main/evaluations/README.md
- TensorFlow Inception graph source: https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py

Download the ImageNet reference archive and TensorFlow Inception graph before running evaluation:

```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb
```

Expected SHA256 checksums:

```text
b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758  VIRTUAL_imagenet256_labeled.npz
009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6  classify_image_graph_def.pb
```

Place both files in the package root:

```text
VIRTUAL_imagenet256_labeled.npz
classify_image_graph_def.pb
```

## Generate 50K Images

Run generation from the package root. The command below uses four Ascend devices, batch size 8, and writes the samples to `eval_outputs/reproduction_50k`.

```bash
ASCEND_VISIBLE_DEVICES=4,5,6,7 \
DEVICES=0,1,2,3 \
MASTER_PORT=49261 \
HCCL_IF_BASE_PORT=61000 \
HCCL_HOST_SOCKET_PORT_RANGE=61000-61099 \
HCCL_NPU_SOCKET_PORT_RANGE=61100-61199 \
NUM_SAMPLES=50000 \
PUBLIC_EVAL_BATCH_SIZE=8 \
OUT_DIR="$PWD/eval_outputs/reproduction_50k" \
bash run_generate.sh
```

Use a different free `MASTER_PORT` and HCCL port range if another distributed job is running on the same machine. The default checkpoint is `global_step1400000`.

The generated PNG files are written under:

```text
eval_outputs/reproduction_50k/cfg0_S_linear_T_0.99_K_100_P_0.8_E_0.99_C_3.0_O_1.0_Cmin0.2_Cmax0.6_SR0.1_GR0.1_EntF_steps250
```

The sampler uses the released settings: linear schedule, 250 SDE steps, temperature 0.99, top-k 100, top-p 0.8, eta 0.99, classifier-free guidance scale 3.0, `Cmin=0.2`, `Cmax=0.6`, `SR=0.1`, and `GR=0.1`.

## Evaluate With TensorFlow on CPU

The reported result below is produced with the TensorFlow evaluator on CPU:

```bash
SAMPLE_DIR="$PWD/eval_outputs/reproduction_50k/cfg0_S_linear_T_0.99_K_100_P_0.8_E_0.99_C_3.0_O_1.0_Cmin0.2_Cmax0.6_SR0.1_GR0.1_EntF_steps250" \
MAX_IMAGES=50000 \
EVAL_BACKEND=tf \
EVAL_DEVICE=cpu \
PYTHON_BIN=python3 \
bash run_evaluation.sh
```

The evaluator first packages the PNG files into:

```text
eval_outputs/reproduction_50k/cfg0_S_linear_T_0.99_K_100_P_0.8_E_0.99_C_3.0_O_1.0_Cmin0.2_Cmax0.6_SR0.1_GR0.1_EntF_steps250.npz
```

It then writes the metric report to:

```text
eval_outputs/evaluation_summary_tf_cpu.txt
```

Expected result for the released checkpoint and the command above:

```text
Images Count: 50000
FID: 2.0453 | IS: 269.0643
Precision: 0.7801 | Recall: 0.6145
```

The first metric run may create a local reference cache named `VIRTUAL_imagenet256_labeled.npz.fid_stats_cache_tf.pkl`. This cache is optional and can be deleted; it only avoids recomputing reference activations.

## Optional PyTorch Metrics

The PyTorch backend can be run with:

```bash
SAMPLE_DIR="$PWD/eval_outputs/reproduction_50k/cfg0_S_linear_T_0.99_K_100_P_0.8_E_0.99_C_3.0_O_1.0_Cmin0.2_Cmax0.6_SR0.1_GR0.1_EntF_steps250" \
MAX_IMAGES=50000 \
EVAL_BACKEND=torch \
bash run_evaluation.sh
```

The TensorFlow CPU command above is the reference command for the reported FID.
