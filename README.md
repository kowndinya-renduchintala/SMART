# SMART
SMART: Submodular Data Mixture Strategy for Instruction Tuning

This contains the anonymous code for the paper "SMART: Submodular Data Mixture Strategy for Instruction Tuning".

Dockerfile contains all the necessary dependencies to run the code.

The dataset script will download FLAN 2022 dataset and upload SMART data mixtures to the Huggingface Hub and to respect the anonymity policy, we will not be able to provide the HUB_TOKEN and HUB_USERNAME, immediately. We will provide the token and username after the review process.

The same dockerfile can be used to both generate the data mixtures and to train the model - You only need to uncomment the last line in the Dockerfile accordingly.