# MULTILAYER-PERCEPTROM

## Description

The goal of this project is to detect whether a tumor is benign or malignant


### Neural Network

In machine learning, the perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.[1] It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

![Image of Neural Network](https://www.researchgate.net/profile/Facundo_Bre/publication/321259051/figure/fig1/AS:614329250496529@1523478915726/Artificial-neural-network-architecture-ANN-i-h-1-h-2-h-n-o.png)



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
