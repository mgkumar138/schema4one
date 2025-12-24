#!/bin/bash

nrnn_values=(1024 256)
glr_values=(1e-4)
use_stochlearn_values=(false true)
nonlinearity_values=(relu phia tanh)
num_seeds=30

for nrnn in "${nrnn_values[@]}"; do
  for glr in "${glr_values[@]}"; do
    for use_stochlearn in "${use_stochlearn_values[@]}"; do
      for nonlinearity in "${nonlinearity_values[@]}"; do

        job_name="assoc_ff_nrnn${nrnn}_glr${glr}_stoch${use_stochlearn}_nl${nonlinearity}"

cat <<EOT > temp_slurm_${job_name}.sh
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH -p seas_compute
#SBATCH --gres=gpu:0
#SBATCH --mem=5G
#SBATCH -o log_ff/${job_name}_%A_%a.log
#SBATCH -e log_ff/${job_name}_%A_%a.log
#SBATCH --array=0-$((${num_seeds}-1))

eval "\$(conda shell.bash hook)"
conda activate tf

# Use SLURM_ARRAY_TASK_ID as the seed
CMD="python -u ff_cue_coord_server.py \
  --prefix 251225 \
  --nrnn ${nrnn} \
  --glr ${glr} \
  --nonlinearity ${nonlinearity} \
  --seed \${SLURM_ARRAY_TASK_ID}"

# NOTE: this assumes that --use_stochlearn is a boolean flag (store_true)
if [ "${use_stochlearn}" = "true" ]; then
  CMD="\$CMD --use_stochlearn"
fi

echo "Running: \$CMD"
eval "\$CMD"
EOT

        sbatch temp_slurm_${job_name}.sh
        rm temp_slurm_${job_name}.sh

      done
    done
  done
done