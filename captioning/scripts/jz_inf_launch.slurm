#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (4 gpus, soit un noeud entier)
#SBATCH --cpus-per-task=10          # nombre de coeurs à réserver, le max = 40 (la mémoire vive sera proportionelle aux nombres de cpus)
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=10:00:00             # temps exécution maximum demande (HH:MM:SS).
#SBATCH --output=runs/%x_%j.out        # nom du fichier de sortie
#SBATCH --error=runs/%x_%j.err         # nom du fichier d'erreur (ici commun avec la sortie)

cd $HOME
source .bashrc

module purge
module load openjdk
conda activate pytorch  # ou source env/bin/activate

export PYTHONPATH=$PYTHONPATH:/gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2
export DATA_DIR=/gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/

folder="/gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/saves/ftsbleumeteorbs18lr1e-5"


path="checkpoint_2023-03-08-19:58:44_epoch0it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-08-22:25:20_epoch1it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-00:19:38_epoch2it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-01:58:32_epoch3it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-03:57:46_epoch4it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-05:38:15_epoch5it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-07:14:50_epoch6it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-08:53:23_epoch7it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-12:02:14_epoch8it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-16:12:32_epoch9it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-19:08:14_epoch10it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-21:17:07_epoch11it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-09-23:30:27_epoch12it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-10-02:09:22_epoch13it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5

path="checkpoint_2023-03-10-04:42:50_epoch14it6293bs18_bleu,meteor_.pth"
save_model_path="${folder}/${path}"

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/scripts/test_singlegpu.py \
        --is_end_to_end False --save_model_path $save_model_path \
        --features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5


