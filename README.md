# Cycle-GAN implemented in PyTorch

## setup:
### 1. install repository:
```
git clone https://github.com/theopfr/cycle-gan-pytorch.git
cd cycle-gan-pytorch
```

### 2. install requirements:
Requirements: Python >= 3.7, Pytorch, torchvision, tqdm, numpy
```
pip install -r requirements.txt (TODO)
```

---

## train:

### create dataset:
- create a folder inside ``datasets/`` with a descriptive name
- create two sub-folders ``trainA`` and ``trainB``
- put all the images of each of the two image categories in one of the folders (e.g put all the images of horses in ``trainA`` and all the images of zebras in ``trainB``)
  
You can find datasets [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

---

### configure the training inside ``src/train.py``
Inside the ``config`` dictionary at the end of the file, you have to specify a ``run_name`` and the ``dataset_name`` (the name of the folder).
The ``run_name`` will be used to create a folder inside ``runs/`` which will contain the model-weights, train-history and generated images. When you set ``resume`` to ``False``, this folder will be deleted and recreated. Therefore, if you interrup the models training and want to continue later, you have to set ``resume`` to ``True`` or else your progress will be lost.

### all the config keys:
TODO