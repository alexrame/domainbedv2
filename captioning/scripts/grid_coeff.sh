for coeff in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    DATA_DIR=/data/rame/ExpansionNet_v2/github_ignore_material/ python test_singlegpu.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/model_meteor_bs18lr5e6_epoch4.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/model_bleu_bs12lr5e6_epoch4.pth --ensemble wa --coeffs [$coeff]
done


