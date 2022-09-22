from datetime import datetime
import subprocess
import time
import os
import random

IS_WW = True
WW_CPU = "--cpuset-cpus=10,11,12,13,14"
WW_GPU = "NV_GPU=0 "  # Add space at the end

SCREEN_NAME = "test"
USER = subprocess.check_output('echo $USER', shell=True).decode("utf-8").replace("\n", "")

# You can use RIAD inpainting or CAM + STD_DEEPFILL
RIAD = True
CAM = False
STD_DEEPFILL = False

NEED_FOOLBOX = True

def wait_docker(dockerid, initial_wait=20):
    time.sleep(initial_wait)
    s = subprocess.check_output('docker ps', shell=True)
    while USER + "_" + str(dockerid) in str(s):
        time.sleep(140)
        s = subprocess.check_output('docker ps', shell=True)
    return


def add_ww_hw(arg):
    if not IS_WW:
        return ""
    return WW_GPU if "gpu" in arg else WW_CPU


def add_ww_params(exp_id):
    return "--name ${{USER}}_" + str(exp_id) + (" -ti --user $(id -u):$(id -g)" if IS_WW else ' --gpus all --network "host"')


def ww_cmd():
    return "nvidia-docker" if IS_WW else "docker"

# Be sure to edit this absolute path according to needs
HOME_PATH = "/home/" + USER + "/"
PROJECT_PATH = HOME_PATH + "Trust-No-Pixel/"
exp_id = random.randint(100000, 999999)

BASE_CMD = add_ww_hw("gpu") + ww_cmd() + ' run ' + add_ww_params(exp_id) + ' -v ' + PROJECT_PATH + \
           ':/opt/project/ -e PYTHONPATH=/opt/project -e PYTHONUNBUFFERED=1 -e PATH=/opt/conda/envs/py36/bin:/opt' +\
           '/conda/envs:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr' +\
           '/sbin:/usr/bin:/sbin:/bin --rm --runtime=nvidia ' + add_ww_hw("cpu") + \
           ' --workdir=/opt/project ughini/dl:{} python -u /opt/project/{}'

tf2 = "cuda112_tf2_cudnn81_py37"
tf1 = "tensorflow-gpu-1.15"

print("Ready? Press enter.", end='')
input()
print("Go.")

ATTACKS = ["DeepFool"]
NETS = ["Inception v3"]
GANS = ["DeepFill v1"]

for attack in ATTACKS:
    for net in NETS:
        firstTimeGAN = True
        for gan in GANS:

            config_file = open(PROJECT_PATH + "config.py", "r")
            list_of_lines = config_file.readlines()
            writelist = [item if "ATTACK_NAME = " not in item else 'ATTACK_NAME = "' + attack + '"\n' for item in
                         list_of_lines]
            writelist1 = [item if "MODEL_NAME = " not in item else 'MODEL_NAME = "' + net + '"\n' for item in writelist]
            writelist2 = [item if "GAN_TYPE = " not in item else 'GAN_TYPE = "' + gan + '"\n' for item in writelist1]
            config_file.close()
            os.remove(PROJECT_PATH + "config.py")
            config_file = open(PROJECT_PATH + "config.py", "a")
            config_file.writelines(writelist2)
            config_file.close()
            time.sleep(5)

            print(attack + ", " + net + ", " + gan)

            if NEED_FOOLBOX and firstTimeGAN:
                wait_docker(exp_id, 1)
                # sudo -u azureuser screen -S ecc ecc...
                subprocess.check_output("screen -S " + SCREEN_NAME + " -X stuff '" +
                                        BASE_CMD.format(tf2, "1_foolbox_attack.py") +
                                        "'`echo '\015'`", shell=True)
                print("1: " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            elif (not NEED_FOOLBOX) and firstTimeGAN:
                wait_docker(exp_id, 1)
                subprocess.check_output("tar -C " + PROJECT_PATH + "images/adv/ -xzvf " +
                                        HOME_PATH + "adversarial-images/" + attack.replace(" ", "") +
                                        '_' + net.replace(" ", "") + ".tar.gz --strip-components 5", shell=True)
                print("1: " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            if firstTimeGAN:
                wait_docker(exp_id, 120)
                subprocess.check_output("screen -S " + SCREEN_NAME + " -X stuff '" +
                                        BASE_CMD.format(tf2,"nn_check_top5.py") + "'`echo '\015'`",shell=True)
                print("NN: " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                wait_docker(exp_id, 120)
                if CAM:
                    subprocess.check_output("screen -S " + SCREEN_NAME + " -X stuff '" +
                                            BASE_CMD.format(tf2,"2_process_images.py") + "'`echo '\015'`", shell=True)
                print("2: " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            wait_docker(exp_id, 120)
            if STD_DEEPFILL:
                subprocess.check_output("screen -S " + SCREEN_NAME + " -X stuff '" +
                                        BASE_CMD.format(tf1, "3_gan_inpainting.py") + "'`echo '\015'`", shell=True)
            elif RIAD:
                subprocess.check_output("screen -S " + SCREEN_NAME + " -X stuff '" +
                                        BASE_CMD.format(tf1, "RIAD_inpainting.py") + "'`echo '\015'`", shell=True)
            print("3: " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            wait_docker(exp_id, 120)
            subprocess.check_output("screen -S " + SCREEN_NAME + " -X stuff '" +
                                    BASE_CMD.format(tf2, "4_denoise_predict.py") + "'`echo '\015'`", shell=True)
            print("4: " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

            if firstTimeGAN:
                q = subprocess.check_output('tar -zcf ' + HOME_PATH + attack.replace(" ", "") + '_' +
                                            net.replace(" ", "") + '.tar.gz ' + PROJECT_PATH + 'images/adv/', shell=True)

            wait_docker(exp_id, 20)

            firstTimeGAN = False
