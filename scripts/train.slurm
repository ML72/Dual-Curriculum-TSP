#!/bin/bash

# Training commands
echo "TRAINING TSP_50 MODEL 1"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --edit_fn global_perturb --hardness_adaptive_percent 25 --ewc_lambda 1 --run_name hac_ewc_curriculum_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_1
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 50 --run_name hac_halfbatch_1

echo "TRAINING TSP_50 MODEL 2"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --edit_fn global_perturb --hardness_adaptive_percent 25 --ewc_lambda 1 --run_name hac_ewc_curriculum_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_2
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 50 --run_name hac_halfbatch_2

echo "TRAINING TSP_50 MODEL 3"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --edit_fn global_perturb --hardness_adaptive_percent 25 --ewc_lambda 1 --run_name hac_ewc_curriculum_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_3
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 50 --run_name hac_halfbatch_3

echo "TRAINING TSP_50 MODEL 4"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --edit_fn global_perturb --hardness_adaptive_percent 25 --ewc_lambda 1 --run_name hac_ewc_curriculum_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_4
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 50 --run_name hac_halfbatch_4

echo "TRAINING TSP_50 MODEL 5"
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --edit_fn global_perturb --hardness_adaptive_percent 25 --ewc_lambda 1 --run_name hac_ewc_curriculum_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --run_name baseline_unif_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 100 --run_name hac_fullbatch_5
python run.py --problem tsp --graph_size 50 --baseline rollout --epoch_size 65536 --batch_size 1024 --n_epochs 151 --checkpoint_epochs 5 --lr_decay 0.98 --bl_warmup_epochs 0 --pretrain_path pretrained/tsp_50 --hardness_adaptive_percent 50 --run_name hac_halfbatch_5
