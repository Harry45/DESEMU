source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo $(for i in $(seq 1 50); do printf "-"; done)

bins=$1
fname_out=$2​
export OMP_NUM_THREADS=4

/usr/local/shared/slurm/bin/srun -N 2 -n 2 --ntasks-per-node 3 -m cyclic --mpi=pmi2 python samplecobaya.py