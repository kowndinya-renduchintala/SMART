import os
import math
import pickle
import submodlib
import numpy as np
import submodlib.functions as submod_fn
from tqdm.auto import tqdm
import time
from datasets import load_dataset
import argparse

parser=argparse.ArgumentParser()
parser.add_argument(
    "--submix",
    type=str,
    required=True,
    default="dialog",
)
args=parser.parse_args()

os.makedirs("fl_ordering", exist_ok=True)

submixture=args.submix
template_types=[
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt"
]

print(f"Loading Embeddings for {submixture}")
embeddings=np.load(f"prompts_embeddings/{submixture}.npy")
print(f"Loading task indices data for {submixture}")
with open(f"task_indices/{submixture}.pkl", "rb") as f:
    task_indices=pickle.load(f)
subsets={}
pbar=tqdm(range(len(task_indices.keys())))
for task in task_indices.keys():
    subsets[task]={}
    for temp_num, template_type in enumerate(template_types):
        if template_type in task_indices[task]:
            indices=task_indices[task][template_type]
            print(f"{task};{template_type};{len(indices)}")
            task_template_embeddings=embeddings[indices]
            # data_sijs=submodlib.helper.create_kernel(X=task_template_embeddings, metric="cosine", method="sklearn")
            submod_obj=submod_fn.facilityLocation.FacilityLocationFunction(
                n=task_template_embeddings.shape[0],
                separate_rep=False,
                mode="dense",
                data=task_template_embeddings,
                create_dense_cpp_kernel_in_python=False,
                # sijs=data_sijs,
            )
            greedyList=submod_obj.maximize(
                budget=len(task_indices[task][template_type])-1,
                optimizer="LazyGreedy",
                show_progress=True
            )
            subsets[task][template_type]=[(indices[idx], gain) for idx, gain in greedyList]
    pbar.update(1)
with open(f"fl_ordering/{submixture}.pkl", "wb") as f:
    pickle.dump(subsets, f)