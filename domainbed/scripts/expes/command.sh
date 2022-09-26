ssh utr15kn@jean-zay.idris.fr
sbatch -A gtw@v100 enshome3_erm012wn_idn1erm0921r0_lp_0916_r0.slurm

for FILE in enshome_erm0123_lp_0916_r0.slurm_1754883.out enshome_erm0123_lp_0916_r20.slurm_1754895.out
do
        echo $FILE
        echo $FILE >> home0123dnim.py
        cat $FILE | grep printres >> home0123dnim.py
done



sbatch -A gtw@v100 enspacs_erm0123wn_idn1erm0921r0_lp_0916_r20.slurm









