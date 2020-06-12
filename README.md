# MULTILAYER-PERCEPTROM

The goal of this project is to detect if tumor is benign or malignant

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip3 install -r requirements.txt
```

## Usage

```python multilayer_perceptron.py```

```bash
usage: multilayer_perceptron.py [-h] -m {train,predict} -d DATASET
                                [-a {softmax,sigmoid}] [-v]

Breath Cancer Detector

optional arguments:
  -h, --help            show this help message and exit
  -m {train,predict}, --model {train,predict}
                        model wanted
  -d DATASET, --dataset DATASET
                        path of dataset
  -a {softmax,sigmoid}, --activate {softmax,sigmoid}
                        activate function
  -v, --verbose         display each epoch on training

```

## Example

```
python multilayer_perceptron.py -m train -d data.csv -v
python multilayer_perceptron.py -m predict -d data.csv
```
