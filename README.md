Sublementary Code for CoRL Submission
============================

The code given here implements the training and testing of deep regression neural networks.

## Contents
0. [Requirements](#requirements)
0. [Training](#training)
0. [Testing](#testing)

## Requirements
- Install a conda distribution
- Install the conda environment given in environment.yml
  ```bash
  conda env create -f environment.yml
  ```
- Install the python package
  ```bash
  python setup.py install
  ```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset in HDF5 formats, and place them under the `data` folder. The downloading process might take an hour or so. The NYU dataset requires 32G of storage space.
	```bash
	mkdir data
	cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz 
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz 
	cd ..
	```
- Download the [SceneNet](https://robotvault.bitbucket.io/scenenet-rgbd.html) dataset (careful, ~300 GB)
  ```bash
  mkdir data
  cd data
  wget http://www.doc.ic.ac.uk/~ahanda/SceneNetRGBD-train.tar.gz
  wget http://www.doc.ic.ac.uk/~ahanda/SceneNetRGBD-val.tar.gz
  tar -xvf SceneNetRGBD-train.tar.gz
  tar -xvf SceneNetRGBD-val.tar.gz
  cd ..
  ```
- Copy only the relevant part of the [SceneNet](https://robotvault.bitbucket.io/scenenet-rgbd.html) dataset
  (24 images per trajectory) using the `copy_relevant_scenent.py` script.


From here on it is assumed that the torch environment is loaded.
	
## Training
The training scripts come with several options, which can be listed with the `--help` flag.
```bash
python sparse_to_dense/main.py --help
```

To train on the SceneNet data set as in the paper, please use the following options.
```bash
python sparse_to_dense/main.py --data scenenet24 --max-depth 5.0 --num-samples 19200 --sparsifier sim_stereo \
 --use-input --epochs 12 -b 24 -m rgbd --height 240 --width 320 -a resnet50

```


## Testing
To validate the performance of a trained model, run with the `-e` option, along with other model options.
To validate on nyudepthv2 run the following command, where PATH/TO points to the directory created previously containing the trained model in model_best.pth.tar
```bash
python sparse_to_dense/main.py --data nyudepthv2 --resume PATH/TO/model_best.pth.tar --use-input --sparsifier sim_stereo \
  --num-samples 19200 --max-depth 5.0

```

