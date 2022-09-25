ssh utr15kn@jean-zay.idris.fr
sbatch -A gtw@v100 enshome3_erm012wn_idn1erm0921r0_lp_0916_r0.slurm

for FILE in enspacs2_erm013wn_idn1erm0921r0_lp_0916.slurm_1716258.out enspacs2_erm013wn_idn1erm0921r0_lp_0916_r20.slurm_1716263.out enspacs2_erm013wn_idn1erm0921r20_lp_0916.slurm_1716260.out enspacs2_erm013wn_idn1erm0921r20_lp_0916_r20.slurm_1716259.out enspacs2_erm013wn_idn1erm0921r40_lp_0916.slurm_1716262.out enspacs2_erm013wn_idn1erm0921r40_lp_0916_r20.slurm_1716261.out enspacs2_erm013wnr_lp_0906_r0.slurm_1715825.out enspacs2_erm013wnr_lp_0906_r20.slurm_1715826.out
do
        echo $FILE
        echo $FILE >> pacs2dn.py
        cat $FILE | grep printres >> pacs2dn.py
done



