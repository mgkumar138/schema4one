#!/bin/bash

cb_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
usenmc_values=(True False)
num_seeds=100

for cb in "${cb_values[@]}"; do
  for usenmc in "${usenmc_values[@]}"; do

    job_name="sym_cb${cb}_usenmc${usenmc}"

cat <<EOT > temp_slurm_${job_name}.sh
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -p sapphire
#SBATCH --gres=gpu:0
#SBATCH --mem=5G
#SBATCH -o log_obs_sym/${job_name}_%A_%a.log
#SBATCH -e log_obs_sym/${job_name}_%A_%a.log
#SBATCH --array=70-$((${num_seeds}-1))

eval "\$(conda shell.bash hook)"
conda activate tf

CMD="python -u obs_sym_server.py --cb ${cb} --seed \${SLURM_ARRAY_TASK_ID} --usenmc ${usenmc}"

echo "Running: \$CMD"
eval "\$CMD"
EOT

    sbatch temp_slurm_${job_name}.sh
    rm temp_slurm_${job_name}.sh

  done
done