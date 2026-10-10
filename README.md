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

The Caltech101 downloader also fetches `Annotations/`; bounding boxes for saliency localization metrics are derived from those contours.

## Training (Don't run this, I give you the checkpoints)

Supported architectures include ResNets such as `resnet18`, `resnet50`, `resnet101` and ViTs such as `vit_b_16`, `vit_b_32`, `vit_l_16`. Training always starts from torchvision pretrained weights. Checkpoints are saved in `checkpoints/`: the best model plus checkpoints around the best epoch using the configured offsets.
Training defaults live in `config/training.json` (`lr`, `patience`, checkpoint offsets, image size, and dataset split fractions).

Run all listed experiments:

```bash
bash scripts/sh/train_all.sh
```

Or run one training:

```bash
python scripts/py/training_classifier.py --dataset oxford_pet --model resnet50 --output checkpoints/oxford_pet_resnet50.pt
```

## Evaluate classifiers

Evaluate an existing checkpoint:

```bash
python scripts/py/test_classifier.py --checkpoint checkpoints/oxford_pet_resnet50.pt
bash scripts/sh/test_all.sh
```

## Generate saliency map explanations (You can skip)

Saliency maps can be generated with `gradcam`, `guided_gradcam`, `gradcampp`, `xgradcam`, `rise`, or `lime`:

```bash
python scripts/py/generate_saliencies.py oxford_pet gradcam checkpoints/oxford_pet_resnet50.pt
bash scripts/sh/generate_all_saliencies.sh
python scripts/py/evaluate_saliencies.py data/saliencies/oxford_pet/oxford_pet_resnet50/resnet50/gradcam/true
```

## Evaluate saliency map explanations

...

## Uncertainty quantification
