# FairTrade: Fairness in Federated Learning

This repository contains my submission for the **Ensuring Fairness in Federated Learning** challenge from the L3S Research Center, Leibniz University Hannover. The work is based on the provided **FairTrade** codebase and focuses on reproducing, understanding, and extending that implementation rather than rebuilding a federated learning framework from scratch.


## Environment

The experiments were run on **Windows** with **Python 3.10** in a dedicated Conda environment. I used **CPU-only execution**.
Install dependencies (and create a virtual environment) in either of the following ways:
 Option 1: With the .yml file
```powershell
conda env create -f fairness-in-AI.yml
```
or if run from repo root:
```powershell
conda env create -f FairTrade/fairness-in-AI.yml
```
```powershell
conda activate fairness-in-AI
```

Option 2: With requirements.txt
```powershell
pip install -r requirements.txt
```

or if you are at the repo root:
```powershell
pip install -r FairTrade/requirements.txt
```

A few dependency adjustments were needed to make the original repository runnable in this setup. In particular, the original pinned requirements were not fully compatible with my local Windows / Python 3.10 environment, so I used a compatible environment configuration and documented the deviation here for transparency.


## What This Branch Contains

### Task 1
I ran the FairTrade pipeline on the **Adult** dataset using the repository’s default Adult configuration and reproduced plausible results for:
- **accuracy**
- **balanced accuracy**
- **gender fairness** using **Statistical Parity Difference**

The final Task 1 run reported:
- Accuracy: `0.7103`
- Balanced Accuracy: `0.7681`
- Statistical Parity Difference (gender): `0.0226`

### Task 2
I extended the final evaluation to measure **intersectional fairness** on the Adult dataset using:
- gender
- race
- the four intersectional subgroups:
  - White Male
  - White Female
  - Non-White Male
  - Non-White Female

The implementation saves:
- a CSV summary table
- a PNG bar chart


## Reproducibility

The codebase already uses fixed `random_state=42` in multiple data splits, and I have added a fixed seed for repeatable execution. Outputs are written under:

```text
FairTrade/results/adult/
```
For long-running experiments, I would prefer logging the full terminal output to a file.


## How To Run Task 1

From the repository root:

```powershell
cd FairTrade
python FairTrade.py --fairness_notion stat_parity --num_clients 3 --dataset_name adult --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type random
```

To keep a full log while still seeing progress in the terminal:

```powershell
python FairTrade.py --fairness_notion stat_parity --num_clients 3 --dataset_name adult --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type random 2>&1 | Tee-Object task1_run.log
```

This run:
- trains FairTrade on the Adult dataset
- prints accuracy, balanced accuracy, and gender fairness
- saves the Task 1 fairness / accuracy arrays


## How To Run Task 2

Task 2 is integrated into the **final evaluation** of the same Adult run. Use the same command:

```powershell
cd FairTrade
python FairTrade.py --fairness_notion stat_parity --num_clients 3 --dataset_name adult --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type random
```

Or with logging:

```powershell
python FairTrade.py --fairness_notion stat_parity --num_clients 3 --dataset_name adult --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type random 2>&1 | Tee-Object task2_run.log
```

At the end of the run, the code should:
- print the intersectional SPD summary in the terminal
- save:
  - `results/adult/3_task2_intersectional_spd.csv`
  - `results/adult/3_task2_intersectional_spd.png`


## Design Choices

For Task 2, I kept the training procedure unchanged and extended only the final fairness evaluation. It was done as such because:
- the task asks to extend the fairness evaluation
- it does not require modifying the training objective
- keeping Task 2 evaluation-only produces a smaller, clearer diff
- it avoids unnecessary overhead inside the already expensive optimization loop

The implementation uses manual metric computation instead of adding a new fairness library, keeping the code closer to the style of the original repository.


## Code Changes

The original FairTrade repository required a few small fixes and extensions before it could be run reliably in my environment.

These include:
- compatibility fixes for device handling
- a `psmpy` API compatibility adjustment
- a fix for the mismatch between `evaluate()` and its caller
- creation of missing output directories before saving results
- the Task 2 intersectional evaluation and output generation
- logging and reproducibility improvements


## Known Issues

- The Adult run takes much longer in practice than the challenge PDF suggests. The main reason is that the provided FairTrade code repeatedly calls `evaluate()` inside the multi-objective Bayesian optimization loop, which greatly increases total runtime.
- BoTorch may emit numerical optimization warnings during execution. In my CPU-based setup, these warnings did not necessarily stop the run, but they are worth documenting.
- Although the laptop used for this submission has an NVIDIA GPU, CPU execution was more reliable for this repository in my environment.


___________________________________________________
# FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning
As Federated Learning (FL) gains prominence in distributed machine learning applications, achieving fairness without compromising predictive performance becomes paramount. The data being gathered from distributed clients in an FL environment often leads to class imbalance. In such scenarios, balanced accuracy rather than accuracy is the true representation of model performance. However, most state-of-the-art fair FL methods report accuracy as the measure of performance,  which can lead to misguided interpretations of the model's effectiveness to mitigate discrimination. To the best of our knowledge, this work presents the first attempt towards achieving Pareto-optimal trade-offs between balanced accuracy and fairness in a federated environment (FairTrade). By utilizing multi-objective optimization, the framework negotiates the intricate balance between model's balanced accuracy and fairness. The framework's agnostic design adeptly accommodates both statistical and causal fairness notions, ensuring its adaptability across diverse FL contexts. We provide empirical evidence of our novel framework's efficacy through extensive experiments on five real-world datasets and comparisons with six competing baselines. The empirical results underscore the significant potential of our framework in improving the trade-off between fairness and balanced accuracy in FL applications.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)
## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process datasets:

- `load_data_utilities.py`: Utility script for loading and preprocessing all the datasets (Adult, Bank, Default, Law).

### Utility Scripts
- `utilities.py`: Utility script for computing evaluation metrics including 'statistical parity', average treatment effect (ATE), balanced accuracy, and accuracy.

### FairTrade main scripts
The following scripts constitute the complete methodology of FairTrade
- `Fairtrade-crypten.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets with secure multiparty protocol.
- `Fairtrade.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets without secure multiparty protocol.

- `constraint.py`: The script contains the implementation of fairness constraints for discrimination mitigation.
  
## Running the FairTrade-crypten.py Script

To run the `FairTrade-crypten.py` script with the default settings, you can use the following command:

```bash
python FairTrade-crypten.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Running the FairTrade.py Script
To run the `FairTrade.py` script with the default settings, you can use the following command:

```bash
python FairTrade.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- torch==2.0.1
- torchvision==0.15.2
- scikit-learn==0.24.2
- pandas==1.5.3
- gpytorch==1.10
- botorch==0.8.5
- crypten==0.4.1
- cvxopt==1.3.1
- cvxpy==1.3.2

## Citation Request
If you find this work useful in your research, please consider citing:
```bash
@inproceedings{badar2024fairtrade,
  title={FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning},
  author={Badar, Maryam and Sikdar, Sandipan and Nejdl, Wolfgang and Fisichella, Marco},
  booktitle={Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
