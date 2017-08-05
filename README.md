# Spatial Pyramid Pooling in Deep Convolutional Networks using tensorflow

## Descriptions
I implemented a [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729) on top of AlexNet in **tensorflow**. Then I applied it to 102 Category Flower identification task.
I implemented for identification task only. If you are interested in this project, I will continue to develop it in object detection task. Do not hesitate to contact me at binhtd.hust@gmail.com. :)

[more information](https://peace195.github.io/spatial-pyramid-pooling/)
## Data

[102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Requirements

* python 2.7
* tensorflow 1.2
* pretrained parameters of AlexNet in ImageNet dataset: [bvlc_alexnet.npy](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) 

## Running
	
	$ python alexnet_spp.py

## Result
82% accuracy rate (the state-of-the-art is 94%).

## Author

**Binh Thanh Do**

