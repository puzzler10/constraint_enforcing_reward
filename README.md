



To run: 

```
python run.py
```

The command accepts flags, like 

```
python run.py --dataset_name=financial --decode_method_eval=sample --gen_train_temperature=0.85 --kl_coef=0.4 --lr=0.0001 --n_eval_seq=8  --seed=1001
```

The `./data/` directory contains a simple dataset that is useful for testing. Other datasets will be downloaded by the script. 






Running baselines: 

```
# Navigate to folder, activate virtual env
conda activate /data/tproth/travis_release/.conda/envs/constrained_adversarial_reward

# Run with flags. 
python -m baselines.py --ds_name rotten_tomatoes 
```