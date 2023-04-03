
export PYTHONPATH=$PYTHONPATH:/home/rame/ExpansionNet_v2

# save_model_path="$@"
# for coeff in -1. -0.5 -0.4 -0.3 -0.2 -0.1 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 2.0
# do
# /home/rame/anaconda3/envs/pytorch/bin/python /home/rame/ExpansionNet_v2/scripts/test_singlegpu.py \
#     --is_end_to_end False --ensemble wa --coeffs [$coeff] --save_model_path $save_model_path \
#     --features_path /data/rame/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5
# done



python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-09-17:18:53_epoch0it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-09-19:08:53_epoch1it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-09-21:01:20_epoch2it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-09-22:54:46_epoch3it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-00:47:56_epoch4it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-02:41:40_epoch5it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-04:35:17_epoch6it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-06:28:54_epoch7it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-08:23:15_epoch8it6293bs18_meteor_.pth
python test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-12:40:59_epoch8it6293bs18_meteor_.pth










