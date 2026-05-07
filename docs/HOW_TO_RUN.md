# How to run

This document walks through the steps required to reproduce the results.

## 1.  Pre-requisites

* A machine with an NVIDIA GPU and CUDA 11.8 or newer (16 GB of VRAM is enough).
  A free Google Colab T4 instance has been used for development.
* Python 3.10+.

## 2.  Get the data

The dataset is not redistributed in this repository.  Place the supplied
`Data.zip` next to the repository, then unpack it:

```bash
unzip Data.zip -d ./Data/
```

After unpacking, the directory should contain:

```
Data/
  frame_000319.png
  frame_000333.png
  ...
  poses.json
```

## 3.  Get the model weights

```bash
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

OWL-ViT weights are downloaded automatically by `transformers` on first run.

## 4.  Install Python packages

```bash
pip install -r requirements.txt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## 5.  Run

### 5a.  As a script

```bash
python -m src.driver --sam-ckpt mobile_sam.pt
```

The result is written to `outputs/answers.json`.  Pass `--fast` for a half-
resolution debug run.

### 5b.  As a notebook

Open `notebook/pipeline.ipynb` in Colab or Jupyter and run the cells from
top to bottom.

## 6.  Adding new entities

Extend the `PROMPTS` dictionary in `src/settings.py`:

```python
PROMPTS["audio_jack"] = "a green or pink 3.5 mm audio jack on a computer back panel"
```

…then re-run the driver.  The new entity is detected, segmented and
back-projected into a 3-D bounding box automatically.
