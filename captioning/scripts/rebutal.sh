#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
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
echo JAVAHOME
echo $JAVAHOME
conda activate pytorch  # ou source env/bin/activate

export PYTHONPATH=$PYTHONPATH:/gpfsdswork/projects/rech/edr/utr15kn/domainbedv2/captioning
export DATA_DIR=/gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/

folder_bleu="/gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/saves/ftsbleubs18lr1e-5"
folder_rouge="/gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/saves/ftsrougebs18lr1e-5"
i=-1
for file_bleu in checkpoint_2023-03-07-11:29:44_epoch0it6293bs18_bleu_.pth checkpoint_2023-03-07-13:03:16_epoch1it6293bs18_bleu_.pth checkpoint_2023-03-07-14:39:43_epoch2it6293bs18_bleu_.pth checkpoint_2023-03-07-16:09:19_epoch3it6293bs18_bleu_.pth checkpoint_2023-03-07-17:41:49_epoch4it6293bs18_bleu_.pth checkpoint_2023-03-07-21:05:09_epoch5it6293bs18_bleu_.pth checkpoint_2023-03-08-11:44:31_epoch6it6293bs18_bleu_.pth checkpoint_2023-03-08-13:11:41_epoch7it6293bs18_bleu_.pth checkpoint_2023-03-08-14:45:00_epoch8it6293bs18_bleu_.pth checkpoint_2023-03-08-17:27:15_epoch9it6293bs18_bleu_.pth
do
((i=i+1))
echo i: $i
j=-1
for file_rouge in checkpoint_2023-03-07-11:29:14_epoch0it6293bs18_rouge_.pth checkpoint_2023-03-07-13:02:15_epoch1it6293bs18_rouge_.pth checkpoint_2023-03-07-14:35:04_epoch2it6293bs18_rouge_.pth checkpoint_2023-03-07-16:09:15_epoch3it6293bs18_rouge_.pth checkpoint_2023-03-07-17:42:17_epoch4it6293bs18_rouge_.pth checkpoint_2023-03-07-21:07:31_epoch5it6293bs18_rouge_.pth checkpoint_2023-03-08-11:45:33_epoch6it6293bs18_rouge_.pth checkpoint_2023-03-08-13:17:12_epoch7it6293bs18_rouge_.pth checkpoint_2023-03-08-14:55:01_epoch8it6293bs18_rouge_.pth checkpoint_2023-03-08-16:40:53_epoch9it6293bs18_rouge_.pth
do
((j=j+1))
echo j: $j
if [[ "$i" == "$j" ]]
then
echo i, j: $file_bleu, $file_rouge
# for coeff in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
# do
# 	/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/domainbedv2/captioning/scripts/test_singlegpu.py \
# 		--is_end_to_end False --ensemble wa --coeffs [$coeff] --save_model_path $folder_bleu/$file_bleu $folder_rouge/$file_rouge \
# 		--features_path /gpfswork/rech/edr/utr15kn/dataplace/ExpansionNet_v2/github_ignore_material/raw_data/features_rf.hdf5
# done
fi
done
done
