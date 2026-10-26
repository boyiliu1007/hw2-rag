#!/usr/bin/env bash
# Rebuild corpus -> chunks -> BM25 + dense indexes, run the coverage checks, then score the best
# system on the dev set and write leaderboard predictions.
#   sbatch --partition=preempt --qos=preempt_qos --gres=gpu:2 scripts/slurm/rebuild_eval.sbatch <tag>
#SBATCH --job-name=hw2-rebuild-eval
#SBATCH --partition=neulab-mini
#SBATCH --qos=neulab_mini_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rebuild_eval_%j.log
set -euo pipefail
TAG=${1:-rebuild}
export HF_HOME=/data/user_data/boyil2/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd /data/user_data/boyil2/hw2-rag
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python

$PY -W ignore scripts/build_corpus.py 2>&1 | grep -v "raw pages:"
$PY -m src.rag.chunking --strategy fixed
$PY -m src.rag.retrieval.sparse --strategy fixed
$PY -m src.rag.retrieval.dense --model bge-base --strategy fixed --batch-size 128
echo "=== coverage"; $PY scripts/answer_coverage.py
$PY scripts/entity_check.py data/eval/leaderboard_queries.json --show none

SYS="--retriever hybrid-rrf:bge-base+ce:bge-reranker-v2-m3 --k 10 --reader Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 2 --style short"
$PY scripts/run_qa.py $SYS --name dev_${TAG}
$PY scripts/run_qa.py $SYS --qa data/eval/leaderboard_queries.json --name lb_${TAG}
# same system, descriptive questions answered in one sentence
$PY scripts/run_qa.py ${SYS/--style short/--style routed} --name dev_${TAG}_routed
$PY scripts/run_qa.py ${SYS/--style short/--style routed} --qa data/eval/leaderboard_queries.json --name lb_${TAG}_routed
for r in dev_${TAG} dev_${TAG}_routed; do $PY scripts/error_report.py runs/qa/$r > /dev/null; done
echo "=== DONE"
