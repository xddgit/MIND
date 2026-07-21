# -*- coding: utf-8 -*-
import os
import argparse
import glob
import numpy as np
import re
import sys 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IS_NPU_AVAILABLE = False
NPU_API_VERSION = None


def configure_tensorflow(device):
    import tensorflow.compat.v1 as tf
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

    config = tf.ConfigProto(allow_soft_placement=True)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["ASCEND_VISIBLE_DEVICES"] = ""
        os.environ["EVAL_DEVICE"] = "cpu"
        config.device_count["GPU"] = 0
        config.gpu_options.allow_growth = False
        print(">>> Using TensorFlow CPU backend")
        return tf, config

    os.environ["EVAL_DEVICE"] = "npu"
    is_npu_available = False
    npu_api_version = None
    try:
        import importlib
        importlib.import_module("npu_bridge.npu_init")
        is_npu_available = True
        npu_api_version = "v1_bridge"
    except ImportError:
        try:
            import npu_device
            npu_device.compat.enable_v1()
            is_npu_available = True
            npu_api_version = "v2_device"
        except ImportError:
            print("NPU TensorFlow bridge not found; using the default GPU or CPU backend.")

    if is_npu_available:
        if npu_api_version == "v1_bridge":
            custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["mix_compile_mode"].b = False
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            print("Enabled the TensorFlow 1 NPU bridge configuration.")
        elif npu_api_version == "v2_device":
            npu_device.compat.v1.npu_config.experimental_options({
                "use_off_line": True,
                "mix_compile_mode": False,
            })
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            print("Enabled the TensorFlow 2 NPU device configuration.")
    else:
        config.gpu_options.allow_growth = True

    return tf, config

from PIL import Image
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor

IM_SIZE = 256

def center_crop_arr(pil_image, image_size):
    if pil_image.size == (image_size, image_size):
        return pil_image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y, crop_x = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def _worker_process(img_path):
    try:
        sample_pil = Image.open(img_path).convert("RGB")
        sample_pil = center_crop_arr(sample_pil, IM_SIZE)
        return np.asarray(sample_pil).astype(np.uint8)
    except:
        return None

def H_package_fast(image_folder, output_npz_path, max_images=50000, target_filenames=None):
    if os.path.exists(output_npz_path):
        return True

    imgs_all = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        imgs_all.extend(glob.glob(os.path.join(image_folder, ext)))
    
    if not imgs_all: return False

    if target_filenames is not None:
        imgs_dict = {os.path.basename(p): p for p in imgs_all}
        imgs = [imgs_dict[name] for name in target_filenames if name in imgs_dict]
    else:
        imgs = sorted(imgs_all)
        if max_images > 0 and len(imgs) > max_images:
            imgs = imgs[:max_images]

    if not imgs: return False

    max_workers = min(32, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_worker_process, imgs), total=len(imgs), 
                            desc=f"    Packing {os.path.basename(output_npz_path)}", leave=False))
    
    samples = [r for r in results if r is not None]
    if not samples: return False
    
    print(f"Packed {len(samples)} images into {os.path.basename(output_npz_path)}")
    np.savez(output_npz_path, arr_0=np.stack(samples))
    return True

def get_intersection_filenames(folders):
    sets = []
    for folder in folders:
        f_list = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            f_list.extend([os.path.basename(x) for x in glob.glob(os.path.join(folder, ext))])
        sets.append(set(f_list))
    
    intersection = set.intersection(*sets)
    return sorted(list(intersection))

def get_evaluated_paths(report_path):
    evaluated_paths = set()
    if not os.path.exists(report_path):
        return evaluated_paths
    with open(report_path, "r") as f:
        content = f.read()
        paths = re.findall(r"Full Path:\s*(.*)", content)
        for p in paths:
            evaluated_paths.add(p.strip())
    return evaluated_paths

def main():
    parser = argparse.ArgumentParser(description="Package generated images and evaluate metrics")
    parser.add_argument("--ref_npz", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VIRTUAL_imagenet256_labeled.npz'), help="Reference dataset .npz path")
    parser.add_argument("--root_dir", type=str, required=True, help="Directory for output report and npz files")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use for CUDA-backed evaluation")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "npu"], default="auto", help="Device mode for TensorFlow backend")
    parser.add_argument("--report_name", type=str, default="evaluation_summary.txt", help="Output filename")
    parser.add_argument("--max_images", type=int, default=50000, help="Max images per folder (used in normal/intersect mode)")
    parser.add_argument("--backend", type=str, choices=["torch", "tf"], default="torch", help="Choose the evaluation backend: 'torch' or 'tf'")
    parser.add_argument("--strict_intersect", action="store_true", help="Only package images that exist in ALL subfolders")
    parser.add_argument("--folders", nargs='+', help="Specific generated image folders to evaluate")

    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    if args.backend == "torch":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.report_name in ["evaluation_summary.txt", "evaluation_summary_torch.txt", "evaluation_summary_tf.txt", "evaluation_summary_tf_cpu.txt"]:
        suffix = "tf_cpu" if args.backend == "tf" and args.device == "cpu" else args.backend
        args.report_name = f"evaluation_summary_{suffix}.txt"

    if args.backend == "torch":
        print(">>> Using PyTorch Evaluation Backend")
        from I_evaluator_torch_v3 import Evaluator
        evaluator = Evaluator()
        evaluator.warmup()
    else:
        print(">>> Using TensorFlow Evaluation Backend")
        tf_device = "npu" if args.device == "auto" else args.device
        tf, config = configure_tensorflow(tf_device)
        from I_evaluator import Evaluator as TF_Evaluator
        sess = tf.Session(config=config)
        evaluator = TF_Evaluator(sess)
        evaluator.warmup()

    def _primary_stats(stats_result):
        return stats_result[0] if isinstance(stats_result, (tuple, list)) else stats_result

    cache_path = args.ref_npz + f".fid_stats_cache_{args.backend}.pkl"
    if os.path.exists(cache_path):
        print(f">>> [CACHE] Loading reference FID stats...")
        with open(cache_path, 'rb') as f:
            ref_stats, ref_acts = pickle.load(f)
    else:
        print(f"--- [COMPUTING REFERENCE FID STATS] ---")
        ref_acts = evaluator.read_activations(args.ref_npz)
        ref_stats = _primary_stats(evaluator.read_statistics(args.ref_npz, ref_acts))
        with open(cache_path, 'wb') as f:
            pickle.dump((ref_stats, ref_acts), f)

    report_path = os.path.join(args.root_dir, args.report_name)
    already_done = get_evaluated_paths(report_path)
    eval_files = []

    subfolders = args.folders if args.folders else sorted(glob.glob(os.path.join(args.root_dir, 'cfg*')))

    target_filenames = None
    if args.strict_intersect and len(subfolders) > 1:
        target_filenames = get_intersection_filenames(subfolders)
        if args.max_images > 0: target_filenames = target_filenames[:args.max_images]
        if len(target_filenames) == 0: return

    print(f"Found {len(subfolders)} generated image folders. Filtering {len(already_done)} completed records...")

    for folder in subfolders:
        suffix = "_intersect" if args.strict_intersect else ""
        output_npz = folder.rstrip('/') + suffix + ".npz"
        abs_npz_path = os.path.abspath(output_npz)
        
        if abs_npz_path in already_done:
            continue
        
        has_npz = os.path.exists(abs_npz_path)
        if not has_npz:
            img_count = len(glob.glob(os.path.join(folder, "*.png"))) + len(glob.glob(os.path.join(folder, "*.jpg")))
            if img_count == 0:
                print(f"  - Skip: {os.path.basename(folder)} (Empty: no images and no .npz found)")
                continue

        success = H_package_fast(folder, output_npz, max_images=args.max_images, target_filenames=target_filenames)
        if success: 
            eval_files.append(output_npz)
            print(f"  + Added to evaluation queue: {os.path.basename(output_npz)}")

    if not eval_files:
        print(">>> All valid folders already in report or no valid images found. Nothing to do.")
        return

    print(f"--- [PHASE 2: EVALUATING INDICATORS] ---")
    for sample_npz in tqdm(eval_files, desc="Batch Evaluating"):
        try:
            abs_path = os.path.abspath(sample_npz)
            sample_acts = evaluator.read_activations(sample_npz)
            sample_stats = _primary_stats(evaluator.read_statistics(sample_npz, sample_acts))
            fid = sample_stats.frechet_distance(ref_stats)
            if args.backend == "tf":
                is_score = evaluator.compute_inception_score(sample_acts[0])
                prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
                result_str = (
                    f"{'='*60}\n"
                    f"Full Path: {abs_path}\n"
                    f"Mode: {'STRICT INTERSECTION' if args.strict_intersect else 'NORMAL'}\n"
                    f"Backend: {args.backend.upper()}\n"
                    f"Images Count: {len(sample_acts[0])}\n"
                    f"FID: {fid:.4f} | IS: {is_score:.4f}\n"
                    f"Precision: {prec:.4f} | Recall: {recall:.4f}\n"
                )
            else:
                result_str = (
                    f"{'='*60}\n"
                    f"Full Path: {abs_path}\n"
                    f"Mode: {'STRICT INTERSECTION' if args.strict_intersect else 'NORMAL'}\n"
                    f"Backend: {args.backend.upper()}\n"
                    f"Images Count: {len(sample_acts[0])}\n"
                    f"FID: {fid:.4f}\n"
                )
            print(result_str)
            with open(report_path, "a") as f: f.write(result_str)
        except Exception as e: print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
