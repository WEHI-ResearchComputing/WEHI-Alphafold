#!/bin/bash

#SBATCH --cpus-per-task 20
#SBATCH --mem=400G 
#SBATCH --gres=gpu:A30:1 
#SBATCH --time 48:00:00
#SBATCH -p gpuq
#SBATCH --output=/vast/scratch/users/iskander.j/logs/relax_multi_%a.out 
#SBATCH --job-name=relaxation
#SBATCH --array=0-4


source /stornext/System/data/apps/miniconda3/miniconda3-latest/etc/profile.d/conda.sh 
conda activate alphafold_2.3.0

echo "relaxing "$SLURM_ARRAY_TASK_ID

python3 run_alphafold.py\
        --model_indices=$SLURM_ARRAY_TASK_ID \
        --fasta_paths=/vast/scratch/users/iskander.j/ProteinSeq/Il24-PKR.fasta \
        --output_dir=/vast/scratch/users/iskander.j/nf-af-results_tw --data_dir=/vast/projects/alphafold/databases \
        --uniref90_database_path=/vast/projects/alphafold/databases/uniref90/uniref90.fasta \
        --mgnify_database_path=/vast/projects/alphafold/databases/mgnify/mgy_clusters.fa \
        --template_mmcif_dir=/vast/projects/alphafold/databases/pdb_mmcif/mmcif_files   \
        --max_template_date=2023-09-20 --obsolete_pdbs_path=/vast/projects/alphafold/databases/pdb_mmcif/obsolete.dat \
        --use_gpu_relax=True --bfd_database_path=/vast/projects/alphafold/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --uniref30_database_path=/vast/projects/alphafold/databases/uniref30/UniRef30_2021_03 \
        --model_preset=multimer \
        --pdb_seqres_database_path=/vast/projects/alphafold/databases/pdb_seqres/pdb_seqres.txt \
        --uniprot_database_path=/vast/projects/alphafold/databases/uniprot/uniprot.fasta \
        --num_multimer_predictions_per_model=1  --relax_only 
        
