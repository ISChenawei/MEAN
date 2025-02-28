# MEAN
This repository is the official implementation of the paper "Multi-Level Embedding and Alignment Network with Consistency and Invariance Learning for Cross-View Geo-Localization". 

The current version of the repository can cover the experiments reported in the paper, for researchers in time efficiency. And we will also update this repository for better understanding and clarity.

## 1. For University-1652 dataset.

Train: run *train_university.py*, with --only_test = False.

Test: run *train_university.py*, with --only_test = True, and choose the model in --ckpt_path.


## 2. For SUES-200 dataset.

You need to split the origin dataset into the appropriate format using the script "MEAN-->Corss-MEAN-->dataset-->SUES-200-->split_datasets.py".

The processed format should be:

```
├─ SUES-200
  ├── Training
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
  ├── Testing
    ├── 150/
    ├── 200/ 
    ├── 250/	
    └── 300/
```

The train and test operation is similar to the University-1652 dataset but with the script train_sues200.py

## 4. Models
We provide the trained models in the link below:
[https://drive.google.com/file/d/140xgtckQkRwqgszD1wWiDH6lV7xufa8r/view?usp=drive_link](https://pan.baidu.com/s/1T-W04e_WVGHFEI1IFhQlsQ)

We will update this repository for better clarity ASAP, current version is for quick research for researchers interested in the cross-view geo-localization task.

## 5. Acknowledgement
This repository is built using the Sample4Geo[https://github.com/Skyy93/Sample4Geo] and MCCG[https://github.com/mode-str/crossview] repositories.
