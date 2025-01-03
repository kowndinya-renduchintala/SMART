import os
import pickle
import random

random.seed(69)

SUBMIXTURES=[
    "flan2021",
    "t0",
    "niv2",
    "cot",
    "dialog"
]

template_types=[
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt"
]

os.makedirs("random_ordering", exist_ok=True)

for submixture in SUBMIXTURES:
    with open(f"task_indices/{submixture}.pkl", "rb") as f:
        task_indices=pickle.load(f)

    random_ordering={}

    for task in task_indices.keys():
        random_ordering[task]={}
        for template_type in template_types:
            if template_type in task_indices[task]:
                indices=task_indices[task][template_type]
                random.shuffle(indices)
                random_ordering[task][template_type]=[(idx, 1) for idx in indices]

    with open(f"random_ordering/{submixture}.pkl", "wb") as f:
        pickle.dump(random_ordering, f)