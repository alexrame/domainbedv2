 ssh utr15kn@jean-zay.idris.fr

for FILE in enshome1_erm023wn_idn1erm0921r0_lp_0916_r0.slurm_1667370.out enshome1_erm023wn_idn1erm0921r0_lp_0916_r20.slurm_1667372.out enshome1_erm023wn_idn1erm0921r20_lp_0916_r0.slurm_1667373.out enshome1_erm023wn_idn1erm0921r20_lp_0916_r20.slurm_1667374.out enshome1_erm023wn_idn1erm0921r40_lp_0916_r0.slurm_1667375.out enshome1_erm023wn_idn1erm0921r40_lp_0916_r20.slurm_1667378.out
do
        echo $FILE
        echo $FILE >> home1dn.py
        cat $FILE | grep printres >> home1dn.py
done


