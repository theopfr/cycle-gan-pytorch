
#  🐴🔄🦓 Cycle-GAN implemented in PyTorch

### This repository contains an implementation of the Cylce-GAN architecture as proposed in the [original paper](https://arxiv.org/abs/1703.10593) along with instructions to train on an own dataset.

---

## 👨‍💻 setup:
### 1. install repository:
```
git clone https://github.com/theopfr/cycle-gan-pytorch.git
cd cycle-gan-pytorch
```

### 2. install requirements:
Requirements: ``Python>=3.7``, ``Pytorch``, ``torchvision``, ``tqdm``, ``numpy``
```
pip install -r requirements.txt
```

---

## 🏋️ train:

### 1. create dataset:
- create a folder inside ``datasets/`` with a descriptive name to store your dataset
- create two sub-folders ``trainA`` and ``trainB``
- put all the images of one of the two image categories in one of the folders (e.g put all the images of horses in ``trainA`` and all the images of zebras in ``trainB``)
##### You can find datasets [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

### 2. run the train script:
- navigate to ``src/``
- run the ``train.py`` script by specifying the train arguments and hyperparameters with command line flags (find the train arguments in the table below; the ``run_name`` and ``dataset_name`` flag have to be set)
- example:
    ```
    python .\train.py --run_name "horse-zebra-run" --dataset_name "horse-zebra-dataset" --save_image_intervall 50 --resume False --epochs 100 --image_size 256 --batch_size 1 --num_res_blocks 9 --lr 0.0002 --lr_decay_rate 1 --lr_decay_intervall 200 --gaussian_noise_rate 0.05 --lambda_adversarial 1 --lambda_cycle 10 --lambda_identity 1 
    ```

---

## 🚩 train script flags/arguments:
| argument | type | default | description | 
| :------------- |:-------------:| ----- | ----- |
| ``run_name`` | str | - | Name for the train run (a folder with this name will be created inside ``runs/`` to store train metrics, model checkpoints and generated images). | 
| ``dataset_name`` | str | - | Name of the folder which holds the dataset to train on. | 
| ``resume`` | str | False | Options: "True", "False"; specifies if the train run should be continued if it was previously interrupted (if set to "False", the run-folder will be reinitialized). | 
| ``save_image_intervall`` | int | 50 | Specifies after how many iterations (not epochs!) generated images should be saved to the run-folder. | 
| ``epochs`` | int | 100 | The amount of epochs to train. | 
| ``image_size`` | int | 256 | The image size to which all images with be resized (images will be quadratic). | 
| ``batch_size`` | int | 1 | The batch size. | 
| ``num_res_blocks`` | int | 9 | Amount of residual blocks in the generator model. | 
| ``lr`` | float | 0.0002 | The learning rate. | 
| ``lr_decay_rate`` | float | 1.0 | Decay rate of the learning rate (will be multiplyed to the learning rate, therefore ``1.0`` means no decay). | 
| ``lr_decay_intervall`` | int | 200 | Specifies after how many iterations (not epochs!) the learning rate should be decayed (has to be ``>=1``). | 
| ``gaussian_noise_rate`` | float | 0.05 | Specifies how much gaussian noise will be applied to images before being fed into the discriminator model (will be multiplied with random noise and then added to the images). | 
| ``lambda_adversarial`` | int | 1 | Specifies how much to weight the adverarial loss (will be multiplied with the loss). | 
| ``lambda_cycle`` | int | 10 | Specifies how much to weight the cycle loss (will be multiplied with the loss). | 
| ``lambda_identity`` | int | 1 | Specifies how much to weight the identity loss (will be multiplied with the loss). | 

##### All the default values are chosen as in the original paper to train on the horse-zebra dataset.

---

<!--
```
    datasets/
    |
    |____horse-zebra/
        |
        |____trainA/
        |    |____horse_img1.png
        |    |____. . .
        |
        |____trainB/
             |____zebra_img1.png
             |____. . .
    ```
-->