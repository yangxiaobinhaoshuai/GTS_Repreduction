# Discrete Graph Structure Learning for Forecasting Multiple Time Series

This is a PyTorch implementation of the paper "[Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://openreview.net/pdf?id=WEHSlH5mOk)", ICLR 2021.

## Installation

Install the dependency using the following command:

```bash
pip install -r requirements.txt
```

* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* tensorflow>=1.3.0
* tables
* future


## Data Preparation

The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY) are put into the `data/` folder. They are provided by [DCRNN](https://github.com/chnsh/DCRNN_PyTorch).

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Unzip the datasets
unzip data/metr-la.h5.zip -d data/
unzip data/pems-bay.h5.zip -d data/

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

python -m scripts.generate_training_data --output_dir=data/pems/03 --traffic_df_filename=data/pems/03/PEMS03.npz
python -m scripts.generate_training_data --output_dir=data/pems/04 --traffic_df_filename=data/pems/04/PEMS04.npz --step=12
python -m scripts.generate_training_data --output_dir=data/pems/04 --traffic_df_filename=data/pems/04/PEMS04.npz --step=24
python -m scripts.generate_training_data --output_dir=data/pems/04 --traffic_df_filename=data/pems/04/PEMS04.npz --step=36
python -m scripts.generate_training_data --output_dir=data/pems/04 --traffic_df_filename=data/pems/04/PEMS04.npz --step=48

python -m scripts.generate_training_data --output_dir=data/pems/08 --traffic_df_filename=data/pems/08/PEMS08.npz --step=12
python -m scripts.generate_training_data --output_dir=data/pems/08 --traffic_df_filename=data/pems/08/PEMS08.npz --step=36
python -m scripts.generate_training_data --output_dir=data/pems/08 --traffic_df_filename=data/pems/08/PEMS08.npz --step=48

python -m scripts.generate_training_data --output_dir=data/pems/07 --traffic_df_filename=data/pems/07/PEMS07.npz --step=12
python -m scripts.generate_training_data --output_dir=data/pems/07 --traffic_df_filename=data/pems/07/PEMS07.npz --step=24
python -m scripts.generate_training_data --output_dir=data/pems/07 --traffic_df_filename=data/pems/07/PEMS07.npz --step=36
python -m scripts.generate_training_data --output_dir=data/pems/07 --traffic_df_filename=data/pems/07/PEMS07.npz --step=48
```

## Train Model

When you train the model, you can run:

```bash
# Use METR-LA dataset
python train.py --config_filename=data/model/para_la.yaml --temperature=0.5

# Use PEMS-BAY dataset
python train.py --config_filename=data/model/para_bay.yaml --temperature=0.5

python train.py --config_filename=data/model/para_pems_03.yaml --temperature=0.5

python train.py --config_filename=data/model/para_pems_04.yaml --temperature=0.5

python train.py --config_filename=data/model/para_pems_08.yaml --temperature=0.5

python train.py --config_filename=data/model/para_pems_07.yaml --temperature=0.5
python -m pdb train.py --config_filename=data/model/para_pems_07.yaml --temperature=0.5
```

Hyperparameters can be modified in the `para_la.yaml` and `para_bay.yaml` files.

## Design your own model

You can directly modify the model in the "model/pytorch/model.py" file.

## Citation

If you use this repository, e.g., the code and the datasets, in your research, please cite the following paper:
```
@article{shang2021discrete,
  title={Discrete Graph Structure Learning for Forecasting Multiple Time Series},
  author={Shang, Chao and Chen, Jie and Bi, Jinbo},
  journal={arXiv preprint arXiv:2101.06861},
  year={2021}
}
```

## Acknowledgments

[DCRNN-PyTorch](https://github.com/chnsh/DCRNN_PyTorch), [GCN](https://github.com/tkipf/gcn), [NRI](https://github.com/ethanfetaya/NRI) and [LDS-GNN](https://github.com/lucfra/LDS-GNN).
