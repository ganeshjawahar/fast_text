## Bag of Tricks for Efficient Text Classification

[![Join the chat at https://gitter.im/fast_text/](https://badges.gitter.im/abhshkdz/neural-vqa.svg)](https://gitter.im/fast_text/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

This repository contains the [Torch](http://torch.ch/) implementation of [FastText model](https://arxiv.org/abs/1607.01759). The original C code from Facebook can be found [here](https://github.com/facebookresearch/fastText).

### Performance Comparison - Test accuracy (%) on sentiment datasets
| Code \ Dataset | AG | Sogou | DBP | Yelp P. | Yelp F. | Yah. A. | Amz. F. | Amz. P. |
|---|---|---|---|---|---|---|---|---|
|[Original repo.](https://github.com/facebookresearch/fastText)|92.5|96.8|98.6|95.7|63.9|72.3|60.2|94.6|
|This repo.|92.5|96.8|98.6|95.7|63.9|72.3|60.2|94.6|

### Components
This repository contains the following components:
* `train.lua` - This is a self-contained code to train the FastText model from scratch. Additionally, it has an option to use pre-trained word embeddings, that are specified in [Glove's pre-trained embeddings](http://nlp.stanford.edu/projects/glove/) format.
* `test.lua` - This script helps in testing the trained model with a test dataset.
* `runall.sh` - This script outputs the last column of the Table 1 from the original paper, which is the test accuracy (%) resulting from running FastTex on 8 sentiment datasets.

### Quick Start
Download all the 8 datasets from [Drive](http://goo.gl/JyCnZq) and rename the master directory containing all the tars to 'data'.

To get the performance scores reported in the last column of the Table 1 from the original paper, execute:
```
bash runall.sh
```

### Dependencies
* [Torch](http://torch.ch/)
* xlua
* tds
* cutorch
* cunn
Packages (d) & (e) are required if you want to use GPU.
Packages (b) to (e) can be installed using:
```
luarocks install <package-name>
```

### Options

#### `th train.lua`
* `input`: training file path [data/agnews.train]
* `output`: output file path [agnews.t7]
* `lr`: learning rate [0.05]
* `lrUpdateRate`: change the rate of updates for the learning rate [100]
* `dim`: size of word vectors [10]
* `epoch`: number of epochs [5]
* `wordNgrams`: max length of word ngram [1]
* `seed`: seed for the randum number generator [123]
* `gpu`: whether to use gpu (1 = use gpu, 0 = not) [0]
* `preTrain`: initialize word embeddings with pre-trained vectors? [0]
* `preTrainFile`: file containing the pre-trained word embeddings (should be in http://nlp.stanford.edu/projects/glove/ format). this is valid iff preTrain=1.

#### `th test.lua`
* `model`: trained model file path [agnews.t7]
* `test`: testing file path [data/agnews.test]
* `gpu`: whether to use gpu (1 = use gpu, 0 = not). use the same option used to train the model.

### Known Issues
* Slow: The code is really slow as the implementation for parallel training using threads is not in place.
* Hashing: The logic for hashtrick is not yet implemented.

### Acknowledgements
Many thanks to the contributors of the [FastText project](https://github.com/facebookresearch/fastText) for making their code publicly available as it helped to replicate their setup with ease.

### Author
[Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/)

### Licence
MIT