# deep_learning_for_camera_trap_images
This repository is a fork, please go to [upstream repository](https://github.com/Evolving-AI-Lab/deep_learning_for_camera_trap_images) to see details and the original Readme. New features are build on top of the [fork of Mo-nasr](https://github.com/Mo-nasr/deep_learning_for_camera_trap_images).

Goal: Get this running as a stable API to classify wildlife images. Might add the "animal or no animal in image" classification (phase 1) to the pipeline soon.

----------


## New features

- Using docker-compose to spin up a container
- Using python 3.9
- Converted code from fork using tensorflow 1.14 into tensorflow 2.9 .1
- Super basic API feature to get an image classified using img_path (that is in the same directory right now)
- Csv file with the full species list, taken from the paper appendix
- Csv file with the full activity list (behaviour), taken from the paper


Will try to clean this up and convert more and more eventually.


----------


## Getting started with classification via API (adopted/simplified version Mo-nasr's readme)
**1. Clone the repo**
```
git clone https://github.com/AlexSperka/deep_learning_for_camera_trap_images.git
```

**2. Download the checkpoints from the following link in the same location the repo was cloned**
* Phase 2 (ResNet-152 architecture): https://drive.google.com/file/d/1KTV9dmqkv0xrheIOEkPXbqeg36_rXJ_E/view?usp=sharing

**3. Unzip and paste the checkpoints into working directory, for example by copy/pasting the following commands in the terminal**
```
unzip ./phase2.zip -d ./deep_learning_for_camera_trap_images/phase2/inference/

```
```
cd ./deep_learning_for_camera_trap_images/phase2/

```
**4. Run the docker container**
```
docker-compose up --build

```
**5. Use Postman or any other form to communicate with an API to get image classifications**
- Address: http://localhost:5010/model/api/v1.0/recognize
- Method: POST
- Example Request Body:
```json
{
    "img_path":"images//elephant.JPG"
}
```

**6. Adjust/add configuration in docker-compose.yaml like port mapping and other files as needed**


----------