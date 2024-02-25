
```
# Create new virtual environment, switch to it, install pip for that env 
mamba create --name constrained_adversarial_reward
source activate constrained_adversarial_reward
mamba install -y pip 

pip3 install Cmake
pip3 install numpy pandas jupyter plotly scipy 

pip3 install --upgrade --force-reinstall  torch --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install transformers==4.19.2 datasets==2.4.0 wandb==0.12.10 sentence_transformers==1.1.0 evaluate  
pip3 install   undecorated GPUtil fastcore nltk textattack spacy editdistance textstat psutil lexicalrichness line_profiler jupyter_contrib_nbextensions tensorflow_hub
pip3 install  tqdm hdbscan==0.8.28 umap-learn==0.4.6
# pip3 install tensorflow  # for textattack
pip3 install --force-reinstall protobuf==3.20.1  # there's a conflict, tensorflow needs < 3.20, wandb needs > 3.20 
python -m spacy download en 



# Torch installed with CUDA 11.6
# The conda install seems to install non-cuda torch (running torch.cuda.is_available() needs to give True)
# Sometimes torch gets installed with the non-cuda version with another package, so do force reinstall.


```
