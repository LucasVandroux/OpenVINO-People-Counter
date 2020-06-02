"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# DEBUG
# Get correct Video Codec
if sys.platform == "linux" or sys.platform == "linux2":
    CODEC = 0x00000021
elif sys.platform == "darwin":
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    # The net outputs blob with shape: [N, 5], where N is the number of detected bounding boxes. For each detection, the description has the format: [x_min, y_min, x_max, y_max, conf]
    bboxes = np.reshape(result, (-1, 5)).tolist()
    # print(len(bboxes))

    for bbox in bboxes:
        conf = bbox[4]
        if conf >= args.prob_threshold:
            xmin = int(bbox[0] * width)
            ymin = int(bbox[1] * height)
            xmax = int(bbox[2] * width)
            ymax = int(bbox[3] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            print(f"Drew detection")
    return frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    #Load the model through `infer_network`
    infer_network.load_model(args.model, device = args.device)

    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith(('.jpg', '.bmp', '.png')):
        image_flag = True

    #Handle the input stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input) # TODO check why input is needed again here

    # DEBUG
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    if not image_flag:
        # Create a video writer for the output video
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width,height))
    else:
        out = None

    #Loop until stream is over
    while cap.isOpened():
        #Read from the video capture
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        #Pre-process the image as needed
        net_input_shape = infer_network.get_input_shape()
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        # TODO put this in the Network class
        # TODO try padding instead of resize

        #Start asynchronous inference for specified request
        infer_network.async_inference(p_frame)
        # TODO: see if I can grab another image while this one is getting processed

        # Wait for the result
        if infer_network.wait() == 0:
            # Get the results of the inference request
            result = infer_network.get_output()
            # print(result)

            # DEBUG:
            # Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            if image_flag:
                cv2.imwrite('output_image.jpg', frame)
            else:
                out.write(frame)

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    if not image_flag:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    # client = connect_mqtt()
    client = None
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
