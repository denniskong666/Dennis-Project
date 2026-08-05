# FloodRoad

FloodRoad is an AI image classification project made for the NVIDIA Jetson Orin Nano. It classifies a road as `Flooded` or `Normal`.

Flooded roads can be dangerous after a storm. FloodRoad can help emergency workers and drivers notice roads that may need to be checked first.

The project can classify saved road images or use a USB webcam for live results.

![FloodRoad test result](test_images/floodroad_result.jpg)

## Dataset

The road images came from Kaggle flood and road datasets. I placed the images into `Flooded` and `Normal` classes. I then created balanced `train`, `val`, and `test` folders so the model could learn and be tested fairly.

Dataset references:

- [Roadway Flooding Image Dataset](https://www.kaggle.com/datasets/saurabhshahane/roadway-flooding-image-dataset)
- [Road Condition Alert Dataset](https://www.kaggle.com/datasets/saadkhan0/road-conditionalert-dataset)
- [Flood Classification Dataset](https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset)

The prepared dataset is stored in the FloodRoad data folder inside Jetson Inference.

## Project Files

```text
jetson-inference/
`-- python/
    `-- training/
        `-- classification/
            |-- data/
            |   |-- floodroad/
            |   |   |-- train/
            |   |   |   |-- Flooded/
            |   |   |   `-- Normal/
            |   |   |-- val/
            |   |   |   |-- Flooded/
            |   |   |   `-- Normal/
            |   |   |-- test/
            |   |   |   |-- Flooded/
            |   |   |   `-- Normal/
            |   |   `-- labels.txt
            |   `-- road_camera.py
            |-- models/
            |   `-- floodroad/
            |       |-- tensorboard/
            |       |-- checkpoint.pth.tar
            |       |-- model_best.pth.tar
            |       |-- labels.txt
            |       `-- resnet18.onnx
            |-- train.py
            `-- onnx_export.py
```

## The Algorithm

FloodRoad uses image classification and transfer learning with a ResNet-18 deep neural network. ResNet-18 was already trained to recognize image features such as shapes, colors, and textures. I retrained it with road images so it could recognize flooded and normal roads.

The dataset is divided into three folders:

- `train` teaches the model.
- `val` checks the model during training.
- `test` checks the finished model with images it did not train on.

The `train.py` program trains the model. The `onnx_export.py` program changes the trained model into ONNX format so Jetson Inference can run it. The final model is `models/floodroad/resnet18.onnx`.

For live video, `road_camera.py` reads frames from `/dev/video0`. The model classifies each frame and shows the class name and confidence score. It sends the result to a browser with WebRTC on port `8554`.

## Set Up the Project

The NVIDIA Jetson Orin Nano, JetPack, Python 3, Docker, and a USB webcam are required.

Clone Jetson Inference and this project repository on the Jetson:

```bash
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
```

Copy the project files into Jetson Inference:

```bash
mkdir -p ~/jetson-inference/python/training/classification/models/floodroad
mkdir -p ~/jetson-inference/python/training/classification/data/floodroad

cp ~/Dennis-Project/road_camera.py \
  ~/jetson-inference/python/training/classification/data/road_camera.py

cp ~/Dennis-Project/models/floodroad/resnet18.onnx \
  ~/jetson-inference/python/training/classification/models/floodroad/resnet18.onnx

cp ~/Dennis-Project/models/floodroad/labels.txt \
  ~/jetson-inference/python/training/classification/models/floodroad/labels.txt

cp ~/Dennis-Project/models/floodroad/labels.txt \
  ~/jetson-inference/python/training/classification/data/floodroad/labels.txt
```

## Train the Model

Place the prepared dataset at:

```text
~/jetson-inference/python/training/classification/data/floodroad
```

Start the Jetson Inference Docker container:

```bash
cd ~/jetson-inference
./docker/run.sh
```

Inside Docker, go to the classification folder and train for 50 epochs:

```bash
cd /opt/jetson-inference/python/training/classification

python3 train.py \
  --epochs=50 \
  --model-dir=models/floodroad \
  data/floodroad
```

Export the trained model to ONNX:

```bash
python3 onnx_export.py \
  --model-dir=models/floodroad
```

Check that the model was created:

```bash
ls models/floodroad/resnet18.onnx
```

## Test Saved Images

Run these commands inside the Docker container:

```bash
cd /opt/jetson-inference/python/training/classification

mkdir -p data/floodroad/test_output_flooded
mkdir -p data/floodroad/test_output_normal

imagenet \
  --model=models/floodroad/resnet18.onnx \
  --input_blob=input_0 \
  --output_blob=output_0 \
  --labels=data/floodroad/labels.txt \
  data/floodroad/test/Flooded \
  data/floodroad/test_output_flooded

imagenet \
  --model=models/floodroad/resnet18.onnx \
  --input_blob=input_0 \
  --output_blob=output_0 \
  --labels=data/floodroad/labels.txt \
  data/floodroad/test/Normal \
  data/floodroad/test_output_normal
```

## Run With a Webcam

Connect the USB webcam and run this command inside Docker:

```bash
cd /opt/jetson-inference/python/training/classification

python3 data/road_camera.py \
  --input=/dev/video0 \
  --input-codec=mjpeg \
  --input-width=1280 \
  --input-height=720 \
  --input-rate=30 \
  --output=webrtc://@:8554/road \
  --output-codec=h264 \
  --headless
```

Open `http://JETSON_IP:8554` in Google Chrome. Replace `JETSON_IP` with the Jetson's IP address. Press `Ctrl+C` in the terminal to stop the program.

If Chrome cannot resolve a random `.local` address, open `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns`. Set **Anonymize local IPs exposed by WebRTC** to **Disabled**, and relaunch Chrome.

## Results and Limits

FloodRoad works well on many road images, but it is not perfect. It may have trouble when a normal road is wet, when flood water is hard to see, or when the camera view does not mainly show a road. The confidence score shows how sure the model is about its answer, but it does not prove that the answer is correct.

FloodRoad should help a person notice a possible danger. It should not replace emergency workers, road closures, or official safety warnings.

## Video Demonstration

[View the final video demonstration here]("C:\Users\Student\Desktop\video_demonstration.mp4")

## References

- [Jetson Inference](https://github.com/dusty-nv/jetson-inference)
- iD Tech Gameplan: NVIDIA AI and Machine Learning Academy
- Kaggle datasets listed above
