#!/usr/bin/env python3

import argparse
import sys

import jetson_inference
import jetson_utils


parser = argparse.ArgumentParser(description="FloodRoad webcam classifier")
parser.add_argument("--input", default="/dev/video0")
parser.add_argument("--output", default="display://0")
args, _ = parser.parse_known_args()

network = jetson_inference.imageNet(
    model="/opt/jetson-inference/python/training/classification/models/floodroad/resnet18.onnx",
    labels="/opt/jetson-inference/python/training/classification/data/floodroad/labels.txt",
    input_blob="input_0",
    output_blob="output_0",
)
camera = jetson_utils.videoSource(args.input, argv=sys.argv)
display = jetson_utils.videoOutput(args.output, argv=sys.argv)
font = jetson_utils.cudaFont()

while True:
    image = camera.Capture()
    if image is None:
        continue

    class_id, confidence = network.Classify(image)
    label = network.GetClassLabel(class_id)
    text = f"FloodRoad: {label} ({confidence * 100:.1f}%)"

    font.OverlayText(
        image,
        image.width,
        image.height,
        text,
        5,
        5,
        font.White,
        font.Gray40,
    )
    display.Render(image)
    display.SetStatus(text)
