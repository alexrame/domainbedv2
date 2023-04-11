Command CLI

et devfair:8080 -t "1234:22,18888:8888" --jport 8080


rsync -avz rame@gpu.scai.sorbonne-universite.fr:/data/rame/ermdomainnet0425 .

ssh rame@gpu.scai.sorbonne-universite.fr

Leh9mppzz9

SSID        ISIR
PASS       ISIR2023

ssh alexandrerame@100.96.182.81

module load anaconda3
 source activate pytorch



export PATH=$PATH:"/Applications/Visual Studio Code.app/Contents/Resources/app/bin"


alias lsr='ls -lahtr'
alias scre="screen -r -D"

SSID        ISIR
PASS       ISIR2023


cd $WORK/domainbedv2

module purge
conda activate bias  # ou source env/bin/activate


# Ceci est le fichier de configuration personnel de Git.
[user]
# Veuillez adapter et décommenter les lignes suivantes :
        name = Alexandre RAME
        email = alexandre.rame.cl@gmail.com
[core]
        editor = vim
[credential]
        helper = store
[alias]
        add-commit = !git add -A && git commit
        ac = !git add . && git commit -m
        acp = !git add . && git commit -m "wip" && git push
        tree = log --graph --decorate --pretty=oneline --abbrev-commit
[pull]
        rebase = false


rsync -avz trl utr15kn@jean-zay.idris.fr:/gpfswork/rech/edr/utr15kn/

# command poure notebook sur sorbonne

ssh -v -L localhost:5000:pas:9000 rame@gpu.scai.sorbonne-universite.fr
srun -N1 --gpus-per-node=1 -t 120 --pty bash
(bias) rame@pas:~$ jupyter-notebook --no-browser --port 9000 --ip 0.0.0.0


# pty on jean zay

srun -A edr@v100 --pty --nodes=1 --ntasks-per-node=1 -C v100-16g --cpus-per-task=10 --gres=gpu:1 --time=2:00:00 --hint=nomultithread bash

# command scai

export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"
jupyter-notebook --ip="*" --no-browser --allow-root --port 2222
ssh -L 2222:192.168.0.100:2222 rame@scai
ssh -L 2222:127.0.0.1:2222 rame@zz

# connect notebook from distance

ssh -t -t rame@scai -L 2223:localhost:2223 ssh daft -L 2223:localhost:2223
jupyter notebook  --NotebookApp.allow_origin='*' --port 2223 --ip="*"  --allow-root  --NotebookApp.password=''  --allow_remote_access=true


# wandb

wandb sync /gpfswork/rech/edr/utr15kn/dataplace/experiments/wandb/wandb/offline-run-20230407_114547-fpaxtlh8
