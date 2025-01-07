### Introduction
Using PointNet for 2D Classification and Segmentation

### Installation

### Usage
To train a model to classify point clouds sampled from 2D shapes:

    python train.py --data **
    # python train.py
    train.py -- vanilla code 原始版本，加入特征：knn、圆环、
    train2.py -- 改成卷即conv2d提特征
    train3.py --dice loss , 阈值0.5
    train4.py --自定义采样器 , 0.9
    train5.py --dice loss , 0.9

Log files and network parameters will be saved to `log` folder in default. 

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir log

After the above training, we can evaluate the model and output some visualizations of the error cases.

    python evaluate.py --visu

If you'd like to prepare your own data, you can refer to some helper functions in `utils/data_prep_util.py` for saving and loading txt files.
