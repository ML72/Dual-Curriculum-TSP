#!/bin/bash

# Eval commands for unif distribution
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_unif50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_unif50_test_seed1234/concorde_costs.pkl -f

# Eval commands for gmm distribution
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_gmm50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_gmm50_test_seed1234/concorde_costs.pkl -f

# Eval commands for poly distribution
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_poly50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_poly50_test_seed1234/concorde_costs.pkl -f

# Eval commands for diag distribution
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_ewc_curriculum_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/baseline_unif_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_fullbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f

python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_1 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_2 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_3 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_4 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
python eval.py data/tsp/tsp_diag50_test_seed1234.pkl --model outputs/tsp_50/hac_halfbatch_5 --all_epochs --decode_strategy greedy --eval_batch_size 1000 --oracle_baseline results/tsp/tsp_diag50_test_seed1234/concorde_costs.pkl -f
