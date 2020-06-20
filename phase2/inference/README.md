# How to use the docker image
**1. Clone the repo**
```
git clone https://github.com/Mo-nasr/deep_learning_for_camera_trap_images.git

```

**2. Download the checkpoints from the following link in the same location the repo was cloned**
* Phase 2 (ResNet-152 architecture):

https://drive.google.com/open?id=15oXo7Zm1N9LXMFg0zuxgRM6FJoF-X2BU

**3. Copy and paste the following commands in the terminal**
```
unzip ./phase2.zip -d ./deep_learning_for_camera_trap_images/phase2/inference/

```
```
cd ./deep_learning_for_camera_trap_images/phase2/

```
**4. Build the docker image**
```
sudo docker build ./inference -t animal_recognition

```
* where animal_recognition is the name of the docker image and can be replaced with any name.

**5. Run the docker image**
```
sudo docker run -p 0.0.0.0:5000:5000 animal_recognition

```
**6. After running the docker image, client.py could be run after adding the path of the image whose inference results are required.**
