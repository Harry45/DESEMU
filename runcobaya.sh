source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
python samplecobaya.py

# addqueue -n 2x4 -s -q cmb -c cobaya_test -m 1 ./runcobaya.sh