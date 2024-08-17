CUDA_VISIBLE_DEVICES=1 python eval.py\
    --weights /home/yzk/lung/CoLF/model_weights/LGGGBM/SNN/s_0_checkpoint.pt\
    --fold 0\
    --save_dir results \
    --dataset_dir /home/yzk/lung/dataset \
    --study LGGGBM\
    --data_mode omic\
    --model_type snn\
    --model_size_omic small\
    --target_gene signatures_rnaseq\
    --n_classes 1

CUDA_VISIBLE_DEVICES=1 python eval.py\
    --weights /home/yzk/lung/CoLF/model_weights/LUAD/AMIL/s_0_checkpoint.pt\
    --fold 0\
    --save_dir results \
    --data_dir /mnt/sdd-1/yzk/datasest/TCGA-LUAD/features\
    --dataset_dir /home/yzk/lung/dataset \
    --study LUAD\
    --data_mode path\
    --model_type amil\
    --n_classes 1

python late_fusion.py\
    --wsi_dir /home/yzk/lung/baseline/results_LGGGBM/AMIL_signatures-rnaseq_nll_surv_a0.0_LGGGBM_gc32\
    --rna_dir /home/yzk/lung/baseline/results_LGGGBM/SNN_signatures-rnaseq_nll_surv_a0.0_omicreg1e-05_LGGGBM_gc32\
    --muti_dir ./late_fusion_SNN-AMIL

python draw_heatmap.py\
    --id TCGA-CJ-5672-01Z-00-DX1.E319BB3C-61C0-4324-A448-57B5EC921C17\
    --weights /home/yzk/lung/baseline/results_KIRC/contrast/AMIL_contrast_KIRC_gc128/s_2_checkpoint.pt\
    --wsi_dir /mnt/sdd-1/yzk/TCGA-KIRC\
    --h5_dir /mnt/sdd-1/yzk/datasest/TCGA-KIRC/features/h5_files