#!/usr/bin/env bash
# CPU-only crawl job (resumable); on completion submits the GPU rebuild/eval job.
#   sbatch scripts/slurm/crawl_more.sbatch <rebuild tag>
#SBATCH --job-name=hw2-crawl
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/crawl_more_%j.log
cd /data/user_data/boyil2/hw2-rag
C=".venv/bin/python -m src.rag.scrape.crawl --only"
$C pittsburghpa_gov > logs/more_city.log 2>&1 &
$C cultural_trust pittsburgh_symphony > logs/more_trust.log 2>&1 &
$C pittsburgh_opera the_frick > logs/more_opera.log 2>&1 &
$C carnegie_museums heinz_history_center > logs/more_museums.log 2>&1 &
$C carnegieart_site food_festivals steelers > logs/more_cmoa_food.log 2>&1 &
$C visitpittsburgh_listings > logs/more_vp.log 2>&1 &
wait
J=$(sbatch --parsable --partition=preempt --qos=preempt_qos --gres=gpu:2 --requeue scripts/slurm/rebuild_eval.sbatch ${1:-v8more})
echo "=== CRAWLS DONE, submitted rebuild job $J"
