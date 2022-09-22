# Trust-No-Pixel

This is the official implementation of the defense against adversarial attack accepted at [IJCNN 2022]().


## Set up and Configuration 
* In order to run the code, you need docker, screen and a machine with an NVIDIA GPU.

* The code leverages two different docker images: one for executing the inpainting procedure (based on tensorflow 1.15) and one for the rest of the code (based on 2.4.1). You can build them using the Dockerfile inside the docker folder. 
You can change the name of the docker images in the controller.py script (tf2 = "cuda112_tf2_cudnn81_py37", tf1 = "tensorflow-gpu-1.15")

* You need to download the Imagenet Validation Dataset, and place it under the root of the project: <br />
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar <br />
mkdir /ILSVRC2012_img_val <br />
tar -xvf ILSVRC2012_img_val.tar -C <project root>/ILSVRC2012_img_val/ <br />

* Pretrained versions of [DeepFill](https://github.com/JiahuiYu/generative_inpainting) (v1, v2) over Places 2 Dataset are already presente under generative_inpainting/model_logs or generative_inpainting_v2/model_logs

* All the essential parameters of the experiment execution should be defined in the config.py file, except for ATTACKS, NETS and GANS variables. These should always be list and defined in the controller in order to overwrite those defined in the config.py.

## Execution

* To run the code we suggest to creaate two screen windows: <br />
screen (create a new screen) <br />
ctrl+a ---> :sessionname test (rename the screen, this name is used in the controller.py SCREEN_NAME variable) <br />
detach without killing <br />
screen <br />
ctrl+a ---> :sessionname controller <br />
detach without killing <br />
To execute a full experiment simply run the controller.py script inside the controller screen.
* Edit HOME_PATH variable in controller.py file according to needs.
* The images to use are listed in filestouse.py. You can edit it according to what images you want to use. The script 0_get_images.py to discover which images are correctly classfied by a model.

## Results

Result logs are printed in the test screen