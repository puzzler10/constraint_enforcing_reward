

This repo contains code for the paper "A Constraint-Enforcing Reward for Adversarial Attacks on Text Classifiers", available [here](https://arxiv.org/abs/2405.11904) with the abstract

> Text classifiers are vulnerable to adversarial examples --- correctly-classified examples that are deliberately transformed to be misclassified while satisfying acceptability constraints. The conventional approach to finding adversarial examples is to define and solve a combinatorial optimisation problem over a space of allowable transformations. While effective, this approach is slow and limited by the choice of transformations. An alternate approach is to directly generate adversarial examples by fine-tuning a pre-trained language model, as is commonly done for other text-to-text tasks. This approach promises to be much quicker and more expressive, but is relatively unexplored. For this reason, in this work we train an encoder-decoder paraphrase model to generate a diverse range of adversarial examples. For training, we adopt a reinforcement learning algorithm and propose a constraint-enforcing reward that promotes the generation of valid adversarial examples. Experimental results over two text classification datasets show that our model has achieved a higher success rate than the original paraphrase model, and overall has proved more effective than other competitive attacks. Finally, we show how key design choices impact the generated examples and discuss the strengths and weaknesses of the proposed approach.

The code is provided here for the paper and is intended as a general guide only. 

## Installation 

To run yourself, create a virtual environment (using whatever tool you prefer) and install the packages at `environment.yml` file, which contains a complete list of packages used in the project. 

For a smaller environment, try starting from a basic python install with the following versions of some key packages:
* python=3.10.9
* pandas==1.5.3
* torch==1.13.1+cu116
* transformers==4.26.0
* wandb==0.12.10
* sentence-transformers==1.1.0
* textattack==0.3.8  (for running the baselines)
* datasets==2.4.0
* evaluate==0.4.0



## Running 
After activating the virtual environment, run the main file with

```
python run.py
```

This will run the file with the default parameters that are specified in the init function of src/Config.py. To adjust any of them, you can either change the parameters in this file, or run with specifying parameters as flags, such as 
```
python run.py --dataset_name=financial --decode_method_eval=sample --gen_train_temperature=0.85 --kl_coef=0.4 --lr=0.0001 --n_eval_seq=8  --seed=1001
```

The code can log to Weights & Biases (wandb). It is disabled by default, which is set by calling `wandb.init(mode="disabled")` in the code. To enable WandB, update the config file (see comments in the file) and set the project and entity variables in the Config `__init__` method. 

Other datasets will be downloaded by the script.  The `./data/` directory contains a simple dataset that is useful for testing. 


To run the baselines, run (for example)
```
python -m baselines.py --ds_name rotten_tomatoes 
```

