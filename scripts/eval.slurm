#!/bin/bash

#SBATCH --job-name=evaluation
#SBATCH --mail-user=ml10872@uw.edu

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
echo "TSP20 dcd_local without local search"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP20 dcd_local without local search with sampling"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local --decode_strategy sample --width 5 --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search with sampling"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline --decode_strategy sample --width 5 --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search with sampling"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local --decode_strategy sample --width 5 --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search with sampling"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline --decode_strategy sample --width 5 --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP20 dcd_local without local search at 500 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local/epoch-500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search at 500 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline/epoch-500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search at 500 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local/epoch-500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search at 500 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline/epoch-500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP20 dcd_local without local search at 1000 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local/epoch-1000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search at 1000 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline/epoch-1000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search at 1000 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local/epoch-1000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search at 1000 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline/epoch-1000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP20 dcd_local without local search at 1500 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local/epoch-1500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search at 1500 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline/epoch-1500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search at 1500 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local/epoch-1500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search at 1500 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline/epoch-1500.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP20 dcd_local without local search at 2000 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/dcd_local/epoch-2000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 baseline without local search at 2000 epochs"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_20/baseline/epoch-2000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 dcd_local with local search at 2000 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/dcd_local/epoch-2000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP50 baseline with local search at 2000 epochs"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_50/baseline/epoch-2000.pt --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP20 using TSP50 model dcd_local without local search"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_50/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP20 using TSP50 model baseline without local search"
python eval.py data/tsp/tsp20_test_seed1234.pkl --model outputs/tsp_50/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP100 using TSP50 model dcd_local with local search"
python eval.py data/tsp/tsp100_test_seed1234.pkl --model outputs/tsp_50/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP100 using TSP50 model baseline with local search"
python eval.py data/tsp/tsp100_test_seed1234.pkl --model outputs/tsp_50/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search

echo "TSP50 using TSP20 model dcd_local without local search"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_20/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP50 using TSP20 model baseline without local search"
python eval.py data/tsp/tsp50_test_seed1234.pkl --model outputs/tsp_20/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance
echo "TSP100 using TSP20 model dcd_local with local search"
python eval.py data/tsp/tsp100_test_seed1234.pkl --model outputs/tsp_20/dcd_local --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
echo "TSP100 using TSP20 model baseline with local search"
python eval.py data/tsp/tsp100_test_seed1234.pkl --model outputs/tsp_20/baseline --decode_strategy greedy --eval_batch_size 1000 --data_equivariance --local_search
