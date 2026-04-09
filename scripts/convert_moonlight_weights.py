import os
import subprocess
import sys

def main():
    # === Configuration ===
    # Path to the HuggingFace checkpoint (Input)
    HF_CHECKPOINT = "/root/.cache/modelscope/hub/models/moonshotai/Moonlight-16B-A3B-Instruct"
    
    # Path where the Megatron-LM checkpoint will be saved (Output)
    # This matches the 'ref_load' path expected by your training script
    SAVE_PATH = "/root/models/Moonlight-16B-A3B-Instruct_torch_dist"
    
    # === Environment Setup ===
    # Add project root and Megatron-LM to PYTHONPATH
    # Assuming this script is located in slime/scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    # Add project root and common Megatron paths
    env["PYTHONPATH"] = f"{project_root}:/root/Megatron-LM:{python_path}"
    
    # === Model Architecture Parameters (from models/moonlight.sh) ===
    MOE_SHARED_EXPERTS = 2
    MOE_FFN_HIDDEN = 1408
    # MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
    MOE_SHARED_EXPERT_INTERMEDIATE_SIZE = MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS
    MOE_ROUTER_TOPK_SCALING_FACTOR = 2.446
    NLAYERS = 27
    FIRST_K_DENSE_REPLACE = 1

    # Generate MOE_LAYER_FREQ: [0, 1, 1, ...]
    # 0 for first K layers, 1 for the rest
    moe_layer_freq_list = []
    for i in range(NLAYERS):
        if i < FIRST_K_DENSE_REPLACE:
            moe_layer_freq_list.append(0)
        else:
            moe_layer_freq_list.append(1)
    
    # Format list as string "[0, 1, ...]"
    MOE_LAYER_FREQ = str(moe_layer_freq_list).replace(" ", "")

    # === Construct Arguments ===
    cmd = [
        "python3",
        os.path.join(script_dir, "run_moonlight_conversion.py"),
        
        # Model Args
        "--disable-bias-linear",
        "--num-layers", str(NLAYERS),
        "--hidden-size", "2048",
        "--ffn-hidden-size", "11264",
        "--num-attention-heads", "16",
        "--kv-channels", "128",
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--norm-epsilon", "1e-5",
        "--rotary-percent", "1.0",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--no-masked-softmax-fusion",
        "--vocab-size", "163840",
        "--multi-latent-attention",
        "--kv-lora-rank", "512",
        "--qk-head-dim", "128",
        "--qk-pos-emb-head-dim", "64",
        "--v-head-dim", "128",
        "--qk-layernorm",
        "--rotary-scaling-factor", "1",
        "--rotary-base", "50000",
        "--mscale", "1.0",
        "--mscale-all-dim", "1.0",
        "--attention-softmax-in-fp32",
        "--no-rope-fusion",
        
        # MoE Args
        "--num-experts", "64",
        "--moe-layer-freq", MOE_LAYER_FREQ,
        "--moe-ffn-hidden-size", str(MOE_FFN_HIDDEN),
        "--moe-router-topk", "6",
        "--moe-shared-expert-intermediate-size", str(MOE_SHARED_EXPERT_INTERMEDIATE_SIZE),
        "--moe-router-pre-softmax",
        "--moe-router-score-function", "sigmoid",
        "--moe-router-enable-expert-bias",
        "--moe-router-load-balancing-type", "seq_aux_loss",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-aux-loss-coeff", "0",
        "--moe-router-bias-update-rate", "0",
        "--moe-router-group-topk", "1",
        "--moe-router-num-groups", "1",
        "--moe-grouped-gemm",
        "--moe-router-topk-scaling-factor", str(MOE_ROUTER_TOPK_SCALING_FACTOR),
        "--moe-token-drop-policy", "probs",
        "--moe-router-dtype", "fp32",
        "--moe-permute-fusion",
        
        # Conversion/IO Args
        "--hf-checkpoint", HF_CHECKPOINT,
        "--save", SAVE_PATH,
        
        # Parallelism settings for the converted checkpoint
        # Using 1 for simplicity and compatibility with single-GPU conversion
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--expert-model-parallel-size", "1",
        
        "--use-cpu-initialization",
        "--micro-batch-size", "1",
        "--save-interval", "1",
    ]

    print("=== Starting Conversion ===")
    print(f"Source: {HF_CHECKPOINT}")
    print(f"Target: {SAVE_PATH}")
    print(f"Command: {' '.join(cmd)}")
    print("===========================")

    try:
        subprocess.run(cmd, env=env, check=True)
        print("\nSUCCESS: Conversion completed successfully.")
        print(f"Checkpoint saved to: {SAVE_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Conversion failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
