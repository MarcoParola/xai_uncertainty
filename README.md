# XAI Uncertainty

Info for Gala: the repo does the following:
- Train classifiers
- Evaluate classification performance
- Generate saliency map explanations
- Compute evaluation metrics for saliency maps
- Compute uncertainty metrics (Gala)
- Run correlation between uncertainty and xai metrics 

Next step could be try to calibrate models and see how explanations change!

## Setup

```bash
git clone https://github.com/MarcoParola/xai_uncertainty.git
cd xai_uncertainty
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

## Datasets

Supported datasets: `oxford_pet`, `caltech101`, `fgvc_aircraft`, `stanford_cars`.

```bash
bash download/download_all_datasets.sh
python test/test_datasets.py # test to check that everything is fine.. otherwise PD
```

## Training (You can skip this, I give you the checkpoints)

Supported architectures include ResNets such as `resnet18`, `resnet50`, `resnet101` and ViTs such as `vit_b_16`, `vit_b_32`, `vit_l_16`. Training always starts from torchvision pretrained weights.
Run all listed experiments:

```bash
bash scripts/sh/train_all.sh
python scripts/py/training_classifier.py --dataset oxford_pet --model resnet50 --output checkpoints/oxford_pet_resnet50.pt # for a single run
```


## Evaluate classifiers (You can skip this, I'll do this)

Evaluate an existing checkpoint:

```bash
python scripts/py/test_classifier.py --checkpoint checkpoints/oxford_pet_resnet50.pt
bash scripts/sh/test_all.sh
```

## Generate saliency map explanations (You can skip this, I'll do this)

Saliency maps can be generated with `gradcam`, `guided_gradcam`, `gradcampp`, `xgradcam`, `rise`, or `lime`; by default each run samples `saliency_generation_limit` random test images with the configured seed.

```bash
bash scripts/sh/generate_all_saliencies.sh
python scripts/py/generate_saliencies.py oxford_pet gradcam checkpoints/oxford_pet_resnet50.pt
```

## Qualitative Plots

Generate one comparison figure per random test image (10 by default):

```bash
env/bin/python scripts/plot/plot_saliencies.py resnet50 oxford_pet
env/bin/python scripts/plot/plot_saliencies.py resnet50 oxford_pet --limit 10 --random-seed 42
```


## Evaluate saliency map explanations (You can skip this, I'll do this)

```bash
python scripts/py/evaluate_saliencies.py data/saliencies/oxford_pet/oxford_pet_resnet50/resnet50/gradcam/true
```


## Uncertainty quantification (TODO GALA)
