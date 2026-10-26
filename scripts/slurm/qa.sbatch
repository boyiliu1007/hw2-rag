#!/usr/bin/env bash
# Usage: sbatch [--gres=gpu:N] scripts/slurm/qa.sbatch "<run_qa.py args>" ["<run_qa.py args>" ...]
# Each quoted argument is one run_qa.py invocation, executed in order in this job.
#SBATCH --job-name=hw2-qa
#SBATCH --partition=neulab-mini
#SBATCH --qos=neulab_mini_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/qa_%j.log
export HF_HOME=/data/user_data/boyil2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd /data/user_data/boyil2/hw2-rag
export PATH="$PWD/.venv/bin:$PATH"   # vLLM JIT-compiles some kernels with ninja
for a in "$@"; do
  echo "=== run_qa.py $a"
  eval .venv/bin/python scripts/run_qa.py $a
done
