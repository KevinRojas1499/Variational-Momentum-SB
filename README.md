<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/score-based-generative-modeling-with-1/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=score-based-generative-modeling-with-1) -->

# <p align="center">Variational Momentum Schrodinger Bridge <br> AISTATS 2025 <p>


## Requirements

Our repo is built built in Python 3.11 using PyTorch. We recommend using the following commands to install the required requirements:
```shell script
conda create -n variational python=3.11
conda activate variational
pip install -r requirements.txt 
``` 
Additionally if you wish to run the time series experiments you can run:

```shell script
pip install -r time_series/requirements.txt 
``` 
## Checkpoints

We provide our pre-trained checkpoint for CIFAR-10 [here](https://drive.google.com/drive/folders/1KYj37DHXIMzc36gqXOyTCuRoD8nOEofm?usp=sharing).

## Training and evaluation

<details><summary>CIFAR-10</summary>

- Training our CIFAR-10 model on a single node with 4 GPUs and batch size 256 can be done using the following command:

```shell script
torchrun --nproc-per-node 4 training.py --dir experiments/cifar/ --batch_size 256 
```
The first time you do multi-gpu training you will have issues with downloading the dataset. To bypass this simply run:
```shell script
python utils/dataset_utils.py --dataset cifar
```

We monitor the training process using wandb, optionally you can disable it by using the `--disable_wandb` flag. To resume training use the `--load_from_ckpt` flag with a path to the snapshot. Other flags can be found in the click header of the training file. 

- Sampling from a model can be done using the command:

```shell script
torchrun --nproc-per-node 1 sampling.py --dir samples --load_from_ckpt path_to_snapshot.pt --num_samples 50176 --batch_size 512
```
- To evaluate FID you can use the command:
```shell script
torchrun --nproc-per-node 1 fid_score.py --path samples_280_75/ --ref_path cifar10 --res 32
```

</details>

<details><summary> Time Series Data</summary>

- Training and evaluation of the time series datasets presented in our paper can be executed through:

```shell script
bash time_series/run_experiments.sh
```
the runs presented here are not meant to produce identical results to those presented in the paper. 


</details>


## Citation
If you find the code useful for your research, please consider citing our AISTATS paper:

```bib
@inproceedings{rojasvariational,
  title={Variational Schr$\backslash$" odinger Momentum Diffusion},
  author={Rojas, Kevin and Tan, Yixin and Tao, Molei and Nevmyvaka, Yuriy and Deng, Wei},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics}
}
```
