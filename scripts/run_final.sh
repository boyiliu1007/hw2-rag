#!/usr/bin/env bash
cd /data/user_data/boyil2/hw2-rag
export HF_HOME=/data/user_data/boyil2/hf_cache VLLM_WORKER_MULTIPROC_METHOD=spawn PATH="$PWD/.venv/bin:$PATH"
R="--retriever hybrid-rrf:bge-base+ce:bge-reranker-v2-m3 --k 10 --reader Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 1 --quantization fp8"
Q=data/eval/private_test.json
i=1
for style in restate routed short; do
    .venv/bin/python scripts/run_qa.py $R --style $style --qa $Q --name test_v8_${style}_fp8 || exit 1
    .venv/bin/python scripts/make_submission.py runs/qa/test_v8_${style}_fp8 --qa $Q \
        --out system_outputs/system_output_$i.json || exit 1
    i=$((i + 1))
done
echo "=== DONE"
