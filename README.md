# Spatial Pyramid Pooling in Deep Convolutional Networks using tensorflow

## New updates
Instead of sppnet, you can use this block of code in Pytorch to train a neural network with variable-sized inputs:

```python
#With these lines of code below, we can memorize the gradient for later updates using pytorch because the
#loss.backward()function accumulates the gradient. After 64 steps, we call optimizer.step() for updating the parameters.
#https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=8, shuffle=False)
for i, (seqs, labels) in enumerate(train_loader):
	...
	loss = criterion(outputs, labels)
	loss.backward()
	if i % 64 == 0 or i == len(train_loader) - 1:
    		optimizer.step()
    		optimizer.zero_grad()
	...
```

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

**Binh Do**

