#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00

#SBATCH --output=out_eval.txt
#SBATCH --error=err_eval.txt



# Setup commands
. ~/.bashrc
cd /gscratch/socialrl/ml10872/tsp/Dual-Curriculum-TSP/
conda activate dcd_tsp

# Evaluation commands
echo "Run on poly distribution"
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/test_dcd --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/test_hac --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f

echo "100 EPOCHS INTO TRAINING"
echo "TSP50 Vanilla"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 DCD"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_dcd/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_dcd/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 HAC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_hac/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_hac/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 DCD + HAC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac/epoch-100.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

echo "200 EPOCHS INTO TRAINING"
echo "TSP50 Vanilla"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 DCD"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_dcd/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_dcd/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 HAC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_hac/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_hac/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 DCD + HAC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac/epoch-200.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

echo "300 EPOCHS INTO TRAINING"
echo "TSP50 Vanilla"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_vanilla/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 DCD"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_dcd/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_dcd/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 HAC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_hac/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_hac/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
echo "TSP50 DCD + HAC"
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/test_dcd_hac/epoch-300.pt --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
