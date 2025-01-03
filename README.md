# SMART

<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/kowndinya-renduchintala/SMART/blob/main/smart_logo.png" width="500" />
        <!-- <img src="smart_logo.png" width="500" /> -->
    </br>
    <br>
        <strong> Submodular Data Mixture Strategy for Instruction Tuning </strong>
    </br>
    <br>
        (Code for reproducing results in our ACL 2024 Paper)
    </br>
</p>

<p align="center">
    <a href="https://github.com/kowndinya-renduchintala/SMART/blob/main/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/kowndinya-renduchintala/SMART?color=blue">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/kowndinya-renduchintala/SMART">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/kowndinya-renduchintala/SMART">
    </a>
</p>

# About SMART

SMART is a novel data mixture strategy for instruction tuning that utilizes a submodular function to assign importance scores to tasks, determine the mixture weights, and also select non-redundant samples from each task.

# How does SMART work?

## Problem Formulation

Consider a collection $`\mathcal{D}=\{\mathcal{T}_1, \dots, \mathcal{T}_M\}`$ of $`M`$ instruction-formatted task datasets where each $`\mathcal{T}_i=\{(prompt_{ij}, response_{ij})\}_{j=1}^{N_{\mathcal{T}_i}}`$ consists of $`N_{\mathcal{T}_i}`$ $(prompt, response)$ pairs. Let $`M' \leq M`$ and $`N' \leq N`$, how do we select a subset of tasks $`\mathcal{D}'=\{\mathcal{T}'_1, \dots, \mathcal{T}'_{M'}\}`$ (where $`\mathcal{D}' \subseteq \mathcal{D}`$), and subsequently $`S=\{S_1, \dots, S_{M'}\}`$ (where $`S_j \subseteq \mathcal{T}'_j`$) and $`\sum_{j=1}^{M'} |S_j|=N'`$ such that efficiently fine-tuning on the subset $`S`$ alone is (nearly) as effective as fine-tuning on the entire collection $`\mathcal{D}`$?

### SMART Stage-1: Weighted Task Subset Selection

In this first stage, given the instruction-tuning dataset $`\mathcal{D}=\{\mathcal{T}_1, \dots, \mathcal{T}_M\}`$, our goal is to find $`\mathcal{D}'=\{\mathcal{T}_{1}', \dots, \mathcal{T}_{M'}'\}`$ where $`\mathcal{D}' \subseteq \mathcal{D}`$, along with the instance budgets, $`\{N_{1}', \dots, N_{M'}'\}`$, such that $`\sum_{j=1}^{M'}|N_{j}'|=N'`$.

If $`f_1`$ is the submodular function that we use in this stage, $`\mathcal{D}'`$ is given by:


```math
\mathcal{D}'=\underset{\substack{X\subseteq \mathcal{D} \\ |X|\le M'}}{\arg \max}{f_1(X)}
```

To find the instance budgets ($`N_{j}'`$ s), we use the second-order Taylor-softmax operation on the value gains obtained from the greedy algorithm, to compute a probability distribution which determines the probability with which instances will be sampled from a given task i.e., if $`\{g_1, \dots g_{M'}\}`$ are the value gains returned by the greedy algorithm, corresponding to the tasks $`\{\mathcal{T}_{1}', \dots, \mathcal{T}_{M'}'\}`$, the instance budgets are given by

```math
N_{j}'=\frac{(1+g_j+0.5g_j^2)}{\sum_{k=1}^{M'}(1+g_k+0.5g_k^2)} \times N'
```

### SMART Stage-2: Instance Subset Selection

In this stage, given the subset of tasks, $`\{\mathcal{T}_{1}', \dots, \mathcal{T}_{M'}'\}`$, and the instance budgets $`\{N_{1}', \dots N_{M'}'\}`$ from the first stage, the goal is to actually select those many samples from each task. If $`f_2`$ is the submodular function used, the final subset $`\mathcal{S}`$ is given by

```math
\mathcal{S}=\bigcup_{j=1}^{M'}\underset{\substack{X_j\subseteq \mathcal{T}_{j}' \\ |X_j|\le N_{j}^{'}}}{\arg \max}{f_2(X_j)}
```

# How to use this repository?

This repository contains the code for reproducing the results in our ACL 2024 paper. 

- ```instruction_tuner.py``` is the general script that can be used to instruction tune a model. One can run it using ```instruction_tuner.sh```, where one needs to set the following parameters:
    - ```DATA_NAME_OR_PATH``` is the name or path of the instruction tuning dataset (in huggingface format) to be used for fine-tuning.
    - ```MODEL_NAME_OR_PATH``` is the name or path of the model to be fine-tuned.
    - ```OUTPUT_DIR``` is the name of the directory where the fine-tuned model will be saved locally as well as on the huggingface hub.
    - ```HUB_TOKEN``` is the huggingface token to upload the model to the huggingface hub. (It will be available in your huggingface account settings - The model will be directly uploaded to your huggingface hub if this is set)
- The data mixtures used in the paper are available in the huggingface hub at ```kowndinya23/flan2022-4096-{M'}-tasks-{f1}-{N'}-instances-{f2}``` where ```{M'}``` is the number of tasks, ```{f1}``` and ```{f2}``` are the submodular functions used in the two stages, and ```{N'}``` is the number of instances.
    - Note that $`M'\in\{8, 16, 32, 64, 128, 256, 512, 1024, 1840\}`$, $`N'\in\{25000, 50000, 100000, 200000, 400000\}`$, and $`f_1, f_2\in\{fl, gc, logdet\}`$.

## Data Generation Scripts

The data generation scripts used for generating the above data mixtures are available in the ```data_generation_scripts``` folder. It can be directly run using ```get_SMART_mixture.sh```. The artifacts generated by the script are also available [here](https://drive.google.com/drive/folders/1dmaboxejSAt52q90pHFzFKyOTELc08du?usp=sharing).

## Dependencies

Dockerfile contains all the necessary dependencies to run the code. The same dockerfile can be used to both generate the data mixtures and to train the model - You only need to uncomment the last line in the Dockerfile accordingly.

# Citation

If you use *SMART* in your research, please cite our ACL 2024 paper :blush: -


[SMART: Submodular Data Mixture Strategy for Instruction Tuning](https://aclanthology.org/2024.findings-acl.766/) (Renduchintala et al., Findings 2024)



```
@inproceedings{renduchintala-etal-2024-smart,
    title = "{SMART}: Submodular Data Mixture Strategy for Instruction Tuning",
    author = "Renduchintala, H S V N S Kowndinya  and
      Bhatia, Sumit  and
      Ramakrishnan, Ganesh",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.766/",
    doi = "10.18653/v1/2024.findings-acl.766",
    pages = "12916--12934",
    abstract = "Instruction Tuning involves finetuning a language model on a collection of instruction-formatted datasets in order to enhance the generalizability of the model to unseen tasks. Studies have shown the importance of balancing different task proportions during finetuning, but finding the right balance remains challenging. Unfortunately, there`s currently no systematic method beyond manual tuning or relying on practitioners' intuition. In this paper, we introduce SMART (Submodular data Mixture strAtegy for instRuction Tuning) {---} a novel data mixture strategy which makes use of a submodular function to assign importance scores to tasks which are then used to determine the mixture weights. Given a fine-tuning budget, SMART redistributes the budget among tasks and selects non-redundant samples from each task. Experimental results demonstrate that SMART significantly outperforms traditional methods such as examples proportional mixing and equal mixing. Furthermore, SMART facilitates the creation of data mixtures based on a few representative subsets of tasks alone and through task pruning analysis, we reveal that in a limited budget setting, allocating budget among a subset of representative tasks yields superior performance compared to distributing the budget among all tasks. The code for reproducing our results is open-sourced at https://github.com/kowndinya-renduchintala/SMART."
}
```

# License

*SMART* is licensed under the MIT License. See [LICENSE](LICENSE) for more information.