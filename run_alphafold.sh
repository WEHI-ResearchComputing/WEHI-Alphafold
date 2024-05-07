#!/bin/bash

# esteva.m@wehi.edu.au

# Defaults
ALPHAFOLD_SIF=/vast/projects/alphafold/alphafold/alphafold-2.3.2/AlphaFold-2.3.2.sif
DATABASES_DIR=/vast/projects/alphafold/databases
DBS_PRESET_FLAG="--bfd_database_path=/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
UNIREF30_FLAG="--uniref30_database_path=/databases/uniref30/UniRef30_2021_03"
PDBSEQRES_FLAG="--pdb_seqres_database_path=/databases/pdb_seqres/pdb_seqres.txt"
PDB70_FLAG="--pdb70_database_path=/databases/pdb70/pdb70"
UNIPROT_FLAG="--uniprot_database_path=/databases/uniprot/uniprot.fasta"
PROKARYOTE_FLAG="--is_prokaryote_list="
NV_FLAG=--nv

printHelpAndExit() {
  echo
  echo -e "Usage: ${0##*/} [OPTION: -g -l -c -d -p -f -m -n -i -b -r -x -u -h] -o <output path> -t <max template date> <input FASTA file(s)>..."
  echo
  echo    "AlphaFold Apptainer  wrapper - Miguel Esteva (esteva.m@wehi.edu.au)"
  echo
  echo    "Apptainer/Singularity is needed to run this script!"
  echo
  echo    "OPTION:"
  echo -e "  -o  output path            [Required] Path to a directory that will store the results. Will be created if non-existent."
  echo -e "  -g  use GPU                Enable (true)/disable (false) the use of GPU (Default: true)."
  echo -e "  -l  GPU(s) to use          Comma separated list of GPU(s) to use e.g. 0,1,2,3. Will be ignored in Slurm."
  echo -e "  -c  custom templates path  Path to a directory containing custom cif templates. (Default: $DATABASES_DIR/pdb_mmcif/mmcif_files)."
  echo -e "  -s  pdb seqres db path     Path to a file containing custom templates. (Default: $DATABASES_DIR/pdb_seqres/pdb_seqres.txt)."
  echo -e "  -d  databases path         Path to the directory containing relevant databases. (Default: $DATABASES_DIR)."
  echo -e "  -p  database preset        Choose db preset model (reduced_dbs|full_dbs). (Default: full_dbs)."
  echo -e "  -f  features only          Stop pipeline once feature extraction is complete and features.pkl is created."
  echo -e "  -m  model preset           Model to use (monomer|monomer_casp14|monomer_ptm|multimer). (Default: monomer)."
  echo -e "  -n  model indices          Model indices to use (comma-separated list 0,...,4. No spaces). (Default: 0)."
  echo -e "  -i  num of predictions     Number of multimer predictions per model (each with a different random seed)"
  echo    "                                 will be generated per model. E.g. if this is 2 and there are 5 models,"
  echo    "                                 then there will be 10 predictions per input. (Default: 5 per model)."
  echo    "                                 Note: this FLAG only applies if model_preset is multimer (-m multimer)."
  echo -e "  -b  benchmark mode         Enable benchmark mode by setting -b. (Default: disabled)."
  echo    "                                 Run multiple JAX model evaluations to obtain a timing that excludes"
  echo    "                                 the compilation time, which should be more indicative."
  echo    "                                 of the time required for inferencing many proteins."
  echo -e "  -t  max template date     [Required] Maximum template release date to consider in YYYY-MM-DD format."
  echo -e "  -r  models to relax       The models to run the final relaxation step on. (all|best|none, default: best)."
  echo    "                                 If 'all', all models are relaxed which is time consuming.."
  echo    "                                 If 'best', only the most confident model is relaxed."
  echo    "                                 If 'none' relaxation is not run (might yield models with stereochemical violations)."
  echo -e "  -x  use CPU relax         Relaxation on CPU (will disable GPU relaxation). Set with no arguments."
  echo    "                                 Relax on GPU can be much faster than CPU, so it is recommended"
  echo    "                                 to set when GPU is not being used or available. (Default: use GPU)."
  echo -e "  -u  use precomputed msas  Whether to read MSAs that have been written to disk. (Default: false)."
  echo -e "  -h  help                  Print this help menu."
  echo
  echo -e "    Environment variables:"
  echo
  echo -e "       AF_UNIREF30_DB: override HHblits uniref30 database. A directory containing UniRef30_YYYY_MM_* files."
  echo -e "       AF_PDB70_DB: override HHsearch pdb70 database. A directory containing pdb70_* files."
  echo
  echo    "For issues with this script, please contact Helpdesk."
  echo    "For issues with AlphaFold, please refer to: https://github.com/deepmind/alphafold/issues"
  echo 
  exit "$1"
}

# Initialise
benchmark=false
features_only=false
input_files=""
use_precomputed_msas=false
run_relax=true
use_gpu_relax=true
custom_templates_path=""
seqres_path=""

while [ $# -gt 0 ]; do
    unset OPTARG
    unset OPTIND
    while getopts :o:g:l:c:s:d:p:fm:n:i:br:xt:uh opt; do
      if [[ ${OPTARG} == -* ]]; then 
		  echo -e "\nArgument missing for -${opt}"
		  printHelpAndExit 1 
	  fi
      case $opt in
        h) printHelpAndExit 0 ;;
        o) output_path=$(echo $OPTARG | sed 's:/*$::') ;;
        g) gpu=$OPTARG; echo "-g $gpu" ;;
        l) gpu_list=$OPTARG ;;
        c) custom_templates_path=$(echo $OPTARG | sed 's:/*$::'); ;;
        s) seqres_path=$(echo $OPTARG | sed 's:/*$::'); ;;
        d) databases_path=$(echo $OPTARG | sed 's:/*$::'); ;;
        p) db_preset=$OPTARG ;;
        f) features_only=true ;;
        m) model_preset=$OPTARG ;;
        n) model_indices=$OPTARG ;;
        i) num_predictions=$OPTARG ;;
        b) benchmark=true ;;
        t) max_template_date=$OPTARG ;;
        r) models_to_relax=$OPTARG ;;
        x) use_gpu_relax=false ;;
        u) use_precomputed_msas=true ;;
        \?) echo "Option $OPTARG not recognised"; printHelpAndExit 1 ;;
        :) echo "Argument missing for -${OPTARG}"
      esac
    done
    shift $((OPTIND-1))
    input_files="${input_files:+${input_files},}${1}"
    shift
done

# Check for Singularity before starting
if ! command -v singularity &> /dev/null; then
	echo -e "\nERROR: Singularity not found on PATH. Singularity"
	echo "       required to run AlphaFold"
	exit 1
fi

# Override HHblits ref database.
if [[ -n $AF_UNIREF30_DB ]]; then
    if [[ -d "$AF_UNIREF30_DB" ]]; then
        SINGULARITY_BIND="${AF_UNIREF30_DB}":/custom/uniref30${SINGULARITY_BIND:+,${SINGULARITY_BIND}}
        UNIREF30_FLAG="--uniref30_database_path=/custom/uniref30/$(basename $AF_UNIREF30_DB)"
    else
        echo -e "\nAF_UNIREF30_DB must be a vailid directory containing UniRef30_YYYY_MM_* files!"
        echo -e "AF_UNIREF30_DB set to $AF_UNIREF30_DB"
        printHelpAndExit 1
    fi
fi

# Override HHsearch ref database.
if [[ -n $AF_PDB70_DB ]]; then
    if [[ -d "$AF_PDB70_DB" ]]; then
         SINGULARITY_BIND="$AF_PDB70_DB":/custom/pdb70${SINGULARITY_BIND:+,${SINGULARITY_BIND}}
         UNIREF30_FLAG="--pdb70_database_path=/custom/pdb70/$(basename $AF_PDB70_DB)"
    else
        echo -e "\nAF_PDB70_DB must be a vailid directory containing pdb70_* files!"
        echo -e "AF_PDB70_DB set to $AF_PDB70_DB"
        printHelpAndExit 1
    fi
fi

if [[ "$custom_templates_path" == "" && "$seqres_path" == "" ]]; then
    # This is the default value if custom templates not set.
    custom_templates_path="/databases/pdb_mmcif/mmcif_files"
elif [[ "$custom_templates_path" != "" && "$seqres_path" == "" ]]; then
        echo -e "\nERROR: -c option set without setting -s option."
        printHelpAndExit 1
elif [[ "$custom_templates_path" == "" && "$seqres_path" != "" ]]; then
        echo -e "\nERROR: -s option set without setting -c option."
        printHelpAndExit 1
else
    if [[ -d "$custom_templates_path" ]]; then
        SINGULARITY_BIND="$custom_templates_path":/custom/templates${SINGULARITY_BIND:+,${SINGULARITY_BIND}}
        custom_templates_path=/custom/templates
    else
        echo -e "\nERROR: $(realpath $custom_templates_path) is not a directory!"
        printHelpAndExit 1
    fi
    if [[ -f "$seqres_path" ]]; then
        SINGULARITY_BIND="$seqres_path":/custom/$(basename "$seqres_path")${SINGULARITY_BIND:+,${SINGULARITY_BIND}}
        PDBSEQRES_FLAG="--pdb_seqres_database_path=/custom/$(basename "$seqres_path")"
    else
        echo -e "\nERROR: $seqres_path is not a valid file."
        printHelpAndExit 1
    fi
fi

if [[ "$input_files" == "" ]]; then
    echo -e "\nERROR: Input file list must be specified."
    printHelpAndExit 1
fi

if [[ "$output_path" == "" ]]; then
    echo -e "\nERROR: Output path directory must be specified."
    printHelpAndExit 1
fi

if [[ "$max_template_date" == "" ]]; then
    echo -e "\nERROR: Max template date is required."
    printHelpAndExit 1
fi

# Check date
if [[ ! "$max_template_date" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]]; then
    echo -e "\nERROR: Invalid date format."
    printHelpAndExit 1
fi

if [[ "$model_preset" == "" ]]; then
    model_preset="monomer"
    echo -e "\nINFO: using -m $model_preset"
fi

if [[ ! "$model_preset" =~ ^(monomer|monomer_casp14|monomer_ptm|multimer)$ ]]; then
    echo -e "\nERROR: invalid option -m $model_preset"
    printHelpAndExit 1
fi

if [[ "$models_to_relax" == "" ]]; then
    models_to_relax="best"
fi

if [[ ! "$models_to_relax" =~ ^(all|best|none)$ ]]; then
    echo -e "\nERROR: invalid option -r $model_preset. Use all, best, or none."
    printHelpAndExit 1
fi

# Arrange databases according to model preset.
if [[ "$model_preset" == "monomer" || "$model_preset" == "monomer_ptm" ]]; then
    PDBSEQRES_FLAG=""
    UNIPROT_FLAG=""
fi

if [[ "$model_indices" == "" ]]; then
    model_indices=0
    echo -e "\nINFO: using -n $model_indices"
fi


if [[ ! "$model_indices" =~ ^([0-4])(,[0-4])?{4}$ ]]; then
    echo -e "\nERROR: invalid -n $model_indices"
	printHelpAndExit 1
fi

if [[  "$databases_path" != "" ]]; then
	echo -e "\nINFO: Setting databases path to $databases_path"
	if [[ ! -d "$databases_path" ]]; then
		echo -e "\nERROR: Invalid databases path $databases_path selected."
		exit 1
	fi
	DATABASES_DIR=$databases_path
fi

# If running on Slurm, Slurm MUST sort GPUS.
if [[ -z $SLURM_JOB_ID ]]; then
    # Start sorting GPUs
    # First: do I have GPUs?
    if [[ $(lspci  | grep -i -c nvidia) -eq 0 ]]; then
    	# No Nvidia GPUs detected
    	echo -e "\nWARNING: No Nvidia GPUs detected. Ignoring GPU options."
    	NV_FLAG=""
        export CUDA_VISIBLE_DEVICES=-1
    else
        # GPUs available. Pay attention to GPU flags.
        if [[ "$gpu" == "" || "$gpu" == true ]]; then
            if [[ ! -z $gpu_list ]]; then
                # No GPU list passed. Use GPU index 0.
                if [[ ! $gpu_list =~ ^[,0-9]+$ ]]; then
                    echo -e "\nERROR: Invalid GPU list $gpu_list"
                    exit 1
                fi
                export CUDA_VISIBLE_DEVICES=$gpu_list
            else
                if command -v nvidia-smi &> /dev/null; then
    		devs=$(nvidia-smi --query-gpu=index --format=csv,noheader)
                    export CUDA_VISIBLE_DEVICES=${devs//$'\n'/,}
                else 
                    # ? Use GPU index 0.
                    export CUDA_VISIBLE_DEVICES=0
                fi 
    	fi
        elif [[ "$gpu" == false || "$features_only" == true ]]; then
             NV_FLAG=""
             export CUDA_VISIBLE_DEVICES=-1
        else
            echo -n "\nERROR: Invalid GPU use option $gpu."
            printHelpAndExit 1
        fi
    fi
else
    if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
        NV_FLAG=""
    fi
fi

# Handle database preset
if [[ "$db_preset" == "" ]]; then
    db_preset="full_dbs"
fi

if [[ ! "$db_preset" =~ (full_dbs|reduced_dbs) ]]; then
    echo -e "\nWARNING: Invalid preset. Only reduced_dbs or full_dbs are valid. Using full_dbs."
    db_preset="full_dbs"
fi

if [[ "$db_preset" == "reduced_dbs" ]]; then
    DBS_PRESET_FLAG="--small_bfd_database_path=/databases/small_bfd/bfd-first_non_consensus_sequences.fasta"
    UNIREF30_FLAG=""
fi

file_index=0

# Process input files into Singularity binds.
while read file; do
    [[ ! -f "$file" ]] && echo -e "\nERROR: $file does not exist." && exit 1
	((file_index++))
    filename=$(basename "$file")
    SINGULARITY_BIND="$file":/input/"$filename"${SINGULARITY_BIND:+,${SINGULARITY_BIND}}
    inputs=/input/${filename}${inputs:+,${inputs}}
done < <(echo "${input_files//,/$'\n'}")
export SINGULARITY_BIND="$SINGULARITY_BIND,$DATABASES_DIR:/databases,$output_path:/output"

# Make additional adjustments if mode is multimer
if [[ "$model_preset" == "multimer" ]]; then
    PDB70_FLAG=""
    if [[ "$num_predictions" == "" ]]; then
        echo -e "\nERROR: Number of predictions not set with -i"
        printHelpAndExit 1
    fi    
    PREDICTIONS="--num_multimer_predictions_per_model=${num_predictions}"
fi

# Check output path
if [[ ! -d "$output_path" ]]; then
   echo -e "\nINFO: Creating $output_path"  
   mkdir -p "$output_path" 
fi

if [[ "`pwd -P | cut -d"/" -f1-3`" == "/stornext/Home" ]]; then 
    echo -e "\nWARNING: Running on /stornext/Home is not recommended."
    echo -e "         Consider vast or HPCScratch instead.\n"
fi

if [[ "`pwd -P | cut -d"/" -f1-3`" == "/stornext/General" ]]; then 
    echo -e "\nWARNING: Running on /stornext/General is not recommended."
    echo -e "         Consider vast or HPCScratch instead.\n"
fi

# Prevent users' python from breaking the container
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# TF/JEX variables:
export TF_FORCE_UNIFIED_MEMORY=1
if [[ -z "$XLA_PYTHON_CLIENT_MEM_FRACTION" ]]; then
	export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
fi

singularity exec $NV_FLAG \
"${ALPHAFOLD_SIF}" /app/run_alphafold.sh \
${DBS_PRESET_FLAG} ${UNIREF30_FLAG} ${PDBSEQRES_FLAG} ${UNIPROT_FLAG} ${PDB70_FLAG} \
--mgnify_database_path=/databases/mgnify/mgy_clusters.fa \
--template_mmcif_dir="$custom_templates_path" \
--obsolete_pdbs_path=/databases/pdb_mmcif/obsolete.dat \
--uniref90_database_path=/databases/uniref90/uniref90.fasta \
--data_dir=/databases \
--output_dir=/output \
--fasta_paths="$inputs" \
${PREDICTIONS} \
--model_indices="$model_indices" \
--max_template_date="$max_template_date" \
--db_preset="$db_preset" \
--model_preset="$model_preset" \
--benchmark=$benchmark \
--use_precomputed_msas=$use_precomputed_msas \
--models_to_relax=$models_to_relax \
--use_gpu_relax=$use_gpu_relax \
--features_only=$features_only \
--logtostderr 2>&1

singularity_exit=$?

if [ $singularity_exit ]; then
	echo -e "\nCOMPLETE: results located at $(realpath $output_path)"
else
	echo -e "\nCOMPLETE: Alphafold Singularity exited with ERROR. Exit: $singularity_exit."
	exit "$singularity_exit"
fi
