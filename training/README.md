
# TESSE Semantic Segmentation Training

Code to train semantic segmentation networks and export trained models as ONNX files. 


## Creating a new segmentation model

- [Collect data](collect-segmentation-data.ipynb): Lauch some TESSE environments from which to collect training data
- [Train a model](train-segmentation-models.ipynb): Once data has been collected, run the provided notebook to train a segmentation network. 


## Installation

This requires Python 3.6+ with the packages listed in [requirements.txt](requirements.txt)

```sh
conda create -n tesse-semantic-segmentation python=3.7 numpy scipy jupyter ipython
conda activate tesse-semantic-segmentation

pip install -r requirements.txt
```
