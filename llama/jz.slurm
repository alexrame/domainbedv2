#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (4 gpus, soit un noeud entier)
#SBATCH --cpus-per-task=10          # nombre de coeurs à réserver, le max = 40 (la mémoire vive sera proportionelle aux nombres de cpus)
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=20:00:00             # temps exécution maximum demande (HH:MM:SS).
#SBATCH --output=runs/%x_%j.out        # nom du fichier de sortie
#SBATCH --error=runs/%x_%j.err         # nom du fichier d'erreur (ici commun avec la sortie)

cd $HOME
source .bashrc

conda activate nlp

export PYTHONPATH=$PYTHONPATH:/gpfsdswork/projects/rech/edr/utr15kn/trl/
export DATA_DIR=/gpfswork/rech/edr/utr15kn/dataplace/experiments/nlp/

export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"
export HF_DATASETS_OFFLINE="1"
export DEBUG="1"
export WANDB_MODE="offline"

# cd /gpfsdswork/projects/rech/edr/utr15kn/trl/
# pip install .

cd /gpfsdswork/projects/rech/edr/utr15kn/trl/examples/llama/
/gpfswork/rech/edr/utr15kn/conda/envs/nlp/bin/python3 llama_inference_review.py --num_samples 200 --peft_names ${DATA_DIR}llama-7b-hf-ppo-imdb-reward-model-dvb-g0-pmedium-03-30-1680170161/epoch191/ ${DATA_DIR}llama-7b-hf-ppo-imdb-reward-model-eld-g0-03-30-1680173234/epoch191/
