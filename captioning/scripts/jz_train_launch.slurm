#!/bin/bash
#SBATCH --partition=gpu_p4
#SBATCH --ntasks=1 \
      # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (4 gpus, soit un noeud entier)
#SBATCH --cpus-per-task=10          # nombre de coeurs à réserver, le max = 40 (la mémoire vive sera proportionelle aux nombres de cpus)
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=20:00:00             # temps exécution maximum demande (HH:MM:SS).
#SBATCH --output=runs/%x_%j.out        # nom du fichier de sortie
#SBATCH --error=runs/%x_%j.err         # nom du fichier d'erreur (ici commun avec la sortie)

cd $HOME
source .bashrc

module purge
module load openjdk
conda activate pytorch  # ou source env/bin/activate

export PYTHONPATH=$PYTHONPATH:/gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2
export DATA_DIR=/gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/

/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/ExpansionNet_v2/train.py\
    --optim_type radam --seed 775533 --sched_type custom_warmup_anneal\
    --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18\
    --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
    --body_save_path ${DATA_DIR}/saves/wa/model_cider.pth --num_epochs 15 \
    --partial_load True --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --save_path ${DATA_DIR}/saves/ftsbleumeteorbs18lr1e-5/ --reinforce bleu,meteor
