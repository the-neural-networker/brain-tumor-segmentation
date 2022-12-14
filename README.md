<div align="center">    
 
# Brain Tumor Segmentation using 3D UNet and UNet Transformers

</div>
 
## Description   
Brain tumor segmentation using 3D UNet and UNet Transformers implemented in Pytorch Lightning. Trained on the [BraTs 2020 dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

## How to run   
First, install dependencies (a new python virtual environment is recommended).   
```bash
# clone project   
git clone https://github.com/visualCalculus/brain-tumor-segmentation

# install project   
cd brain-tumor-segmentation
pip install -r requirements.txt
 ```   
 Next, navigate to src folder and run train.py with appropriate arguments
 ```bash
# module folder
cd src

TRAIN_CSV={path to BraTs 2020 training csv}
TEST_CSV={path to BraTs 2020 testing csv}

# train model
python train.py --gpus 1 --batch_size 1 --max_epochs 50
--train_csv=${TRAIN_CSV} \
--test_csv=${TEST_CSV} \
--model="unetr" \ # unet or unetr
--is_resize true
--learning_rate 5e-4

```

<!-- ## Results
<div align="center">

![result1](misc/result_collage.png)
![result2](misc/result_collage_2.png)
![result3](misc/result_collage_3.png)

</div> -->
