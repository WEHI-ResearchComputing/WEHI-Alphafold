# Utilities 
# Modified For WEHI decoupling only
import numpy as np
import os, json, pickle
import glob
from absl import logging
import matplotlib.pyplot as plt
from alphafold.relax import relax
from alphafold.model import model
from typing import Any, Dict, Mapping, Union


# Chains to/from files
def write_chains(chains_path,chains):
    with open(chains_path, 'w') as filehandle:
        filehandle.writelines("%s\n" % chain for chain in chains)
    
def read_chains(chains_path):
    chains = []
    # open file and read the content in a list
    with open(chains_path, 'r') as filehandle:
        filecontents = filehandle.readlines()
        for line in filecontents:
            current_place = line[:-1]
            chains.append(current_place)
    return chains

# Adding confidence measure pddlt to models and saving to file
def set_bfactor(ip_path,op_path, bfac):
    I = open(ip_path,"r").readlines()
    O = open(op_path,"w")
    for line in I:
        if line[0:6] == "ATOM  ":
            seq_id = int(line[22:26].strip()) - 1
            O.write(f"{line[:60]}{bfac[seq_id]:6.2f}{line[66:]}")
    O.close()
    
# Adding confidence measure pddlt to models, reseting chains, for h>1 only,  and saving to file    
def set_chain_bfactor(ip_path,op_path, bfac,  chains,idx_res=None, is_relaxed=False):

    #logging.info("Chains len : %d",len(chains))
    I = open(ip_path,"r").readlines()
    O = open(op_path,"w")
    for line in I:
        if line[0:6] == "ATOM  ":
          seq_id = int(line[22:26].strip()) - 1
          #logging.info("Seq_id : %d",seq_id)
          if not is_relaxed:
            seq_id = np.where(idx_res == seq_id)[0][0]
          O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
    O.close()
    
##Reranking
def rerank(output_dir:str,
           model_names_extra:list,
           fasta_name:str):
    plddts={}
    for model_name in model_names_extra:
        for filenm in glob.glob(os.path.join(output_dir,f'result_{model_name}.pkl')):
            logging.info(f"Found results for {model_name}")
            result=pickle.load(open(filenm, 'rb'))
            plddts[model_name] = np.mean(result['plddt'])

    ranked_order = []
    logging.info("Ordering and saving.")
    for idx, (model_name, _) in enumerate(sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path=ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
            if os.path.exists(os.path.join(output_dir,f'relaxed_{model_name}.pdb')):
                with open(os.path.join(output_dir,f'relaxed_{model_name}.pdb'),'r') as relaxed_pdbs:
                    f.write(relaxed_pdbs.read())
                    logging.info(f"For {model_name} - relaxed model saved")
            elif os.path.exists(os.path.join(output_dir,f'unrelaxed_{model_name}.pdb')):
                with open(os.path.join(output_dir,f'unrelaxed_{model_name}.pdb'),'r') as unrelaxed_pdbs:
                    f.write(unrelaxed_pdbs.read())
                    logging.info(f"For {model_name} - unrelaxed saved")
    
    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    with open(ranking_output_path, 'a') as f:
        f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))
    
    logging.info("Preparing plots.")
    plots_dir=os.path.join(output_dir,"plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    ranking_dict={'plddts': plddts, 'order': ranked_order}
    feature_dict = pickle.load(open(os.path.join(output_dir, "features.pkl"),'rb'))
    model_dicts={}
    
    for model_name in model_names_extra:
        for filenm in glob.glob(os.path.join(output_dir,f'result_{model_name}.pkl')):
            result=pickle.load(open(filenm, 'rb'))
            model_dicts[model_name] = result
            
    msa = feature_dict['msa']
    seqid = (np.array(msa[0] == msa).mean(-1))
    seqid_sort = seqid.argsort()
    non_gaps = (msa != 21).astype(float)
    non_gaps[non_gaps == 0] = np.nan
    final = non_gaps[seqid_sort] * seqid[seqid_sort, None]
    
    ###################### PLOT MSA WITH COVERAGE ####################
    
    plt.figure(figsize=(14, 4), dpi=100)
    plt.subplot(1, 2, 1)
    plt.title(f"Sequence coverage ({fasta_name})")
    plt.imshow(final,
               interpolation='nearest', aspect='auto',
              cmap="rainbow_r", vmin=0, vmax=1, origin='lower')
    plt.plot((msa != 21).sum(0), color='black')
    plt.xlim(-0.5, msa.shape[1] - 0.5)
    plt.ylim(-0.5, msa.shape[0] - 0.5)
    plt.colorbar(label="Sequence identity to query", )
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
 
    ##################################################################
    ###################### PLOT LDDT PER POSITION ####################
    
    plt.subplot(1, 2, 2)
    plt.title(f"Predicted LDDT per position ({fasta_name})")
    
    s = 0
    for model_name, value in model_dicts.items():
        plt.plot(value["plddt"], 
                 label=f"{model_name}, plddts: {round(list(ranking_dict['plddts'].values())[s], 6)}")
        s += 1
    #plt.legend()
    plt.legend(loc='lower left')
    plt.ylim(0, 100)
    plt.ylabel("Predicted LDDT")
    plt.xlabel("Positions")
    plt.savefig(f"{plots_dir}/{fasta_name}_coverage_LDDT.pdf")
    ##########################################################################################################
    ######## PLOT THE Predicted LDDt and Predeicted Aligned Errorr per model #################################
    if "predicted_aligned_error" in model_dicts[model_names_extra[0]]:
        for n, (model_name, value) in enumerate(model_dicts.items()):
            plt.figure(figsize=[8 * 2, 6],dpi=100)
            plt.subplot(1, 2, 1)
            plt.title(f'Predicted LDDT, {model_name}')
            plt.plot(value["plddt"], 
            label=f"{model_name}, plddts: {round(list(ranking_dict['plddts'].values())[n], 6)}")
            plt.subplot(1, 2, 2)
            plt.title(f'Predicted Aligned Error, {model_name}')
            plt.imshow(value["predicted_aligned_error"], label=model_name, vmin=0., vmax=value["max_predicted_aligned_error"], cmap='Greens_r')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.savefig(f"{plots_dir}/{fasta_name}_PAE_{model_name}.pdf")
    else:
        logging.info("No predicted_aligned_error found. Make sure you choose monomer_ptm when running AlphaFold prediction.")