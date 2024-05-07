#!/bin/bash

#SBATCH --cpus-per-task 20
#SBATCH --mem=50G 
#SBATCH --gres=gpu:P100:1 
#SBATCH --time 48:00:00
#SBATCH -p gpuq
#SBATCH --output=/vast/scratch/users/iskander.j/logs/relax_mono_%a.out 
#SBATCH --job-name=relaxation
#SBATCH --array=0-4


source /stornext/System/data/apps/miniconda3/miniconda3-latest/etc/profile.d/conda.sh 
conda activate alphafold_2.3.0

echo "relaxing "$SLURM_ARRAY_TASK_ID

python3 run_alphafold.py\
        --model_indices=$SLURM_ARRAY_TASK_ID \
        --fasta_paths=/vast/scratch/users/iskander.j/ProteinSeq/q1.fasta \
        --output_dir=/vast/scratch/users/iskander.j/nf-q1 --data_dir=/vast/projects/alphafold/databases \
        --uniref90_database_path=/vast/projects/alphafold/databases/uniref90/uniref90.fasta \
        --mgnify_database_path=/vast/projects/alphafold/databases/mgnify/mgy_clusters.fa \
        --template_mmcif_dir=/vast/projects/alphafold/databases/pdb_mmcif/mmcif_files   \
        --max_template_date=2023-09-20 --obsolete_pdbs_path=/vast/projects/alphafold/databases/pdb_mmcif/obsolete.dat \
        --use_gpu_relax=True --bfd_database_path=/vast/projects/alphafold/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --uniref30_database_path=/vast/projects/alphafold/databases/uniref30/UniRef30_2021_03 \
        --model_preset=monomer_ptm \
        --pdb70_database_path=/vast/projects/alphafold/databases/pdb70/pdb70 \
        --relax_only 
        
        
