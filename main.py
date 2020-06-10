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
from statistics import mean 

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from utils import Tracker, updateTrackers

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
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    # The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
    bboxes = np.reshape(result, (-1, 7)).tolist()
    # print(len(bboxes))

    for bbox in bboxes:
        conf = bbox[2]
        if conf >= args.prob_threshold:
            xmin = int(bbox[3] * width)
            ymin = int(bbox[4] * height)
            xmax = int(bbox[5] * width)
            ymax = int(bbox[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
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
    cap.open(args.input)

    # Grab the shape of the input and the frame rate
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not image_flag:
        # Create a video writer for the output video
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('out.mp4', CODEC, fps, (width,height))
    else:
        out = None

    # Initialize the list of tracked vehicle
    min_frame_count = 3 # minimum number of consecutive frame a pedestrian needs to be detected in 
    list_tracked_pedestrians = []
    list_trackers = [] # List of all trackers
    set_id_pedestrians = set() # Set of all the pedestrians in total
    previous_count = 0

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

        #Start asynchronous inference for specified request
        infer_network.async_inference(p_frame)

        # Wait for the result
        if infer_network.wait() == 0:
            # Get the results of the inference request
            result = infer_network.get_output()

            # Detect the objects in the new frame
            list_detections = infer_network.postprocess_output(result, width, height, args.prob_threshold)

            # Update the position of the tracked pedestrians
            list_trackers, list_detections, list_trackers_removed = updateTrackers(list_trackers, list_detections)        

            # Add the remaining detections as new tracked pedestrians
            for detection in list_detections:
                x, y, w, h = detection
                list_trackers.append(Tracker(x, y, w, h))

            # Get the list of detected pedestrians (trackers detected in more than min_frame_count)
            list_tracked_pedestrians = [tracker for tracker in list_trackers if len(tracker.list_centroids) >= min_frame_count]

            # Draw all the tracked vehicles in the current frame
            for pedestrian in list_tracked_pedestrians:
                pedestrian.drawOnFrame(frame)

            # --- Extract any desired stats from the results ---

            # Update the list of total pedestrians 
            set_id_pedestrians = set_id_pedestrians.union(set([p.id for p in list_tracked_pedestrians]))
            
            # Number of pedestrians in the current frame
            current_count = len(list_tracked_pedestrians)

            # Total of pedestrians detected since the beginning of the video
            total_count = len(set_id_pedestrians)

            # Publish the results in the person topic
            if current_count != previous_count:
                previous_count = current_count
                client.publish("person", json.dumps({"count": current_count, "total":total_count}))

            # Get the total duration a person stayed in the frame when he/she leave the frame
            duration_min = 10 # minimum frame a tracker needs to exist for its duration to be taken in account

            if list_trackers_removed:
                print("Pedestrian left the frame")
                list_duration = [len(p.list_centroids) * 1/fps for p in list_trackers_removed if len(p.list_centroids) > duration_min]
                if list_duration:
                    duration = mean(list_duration)
                    client.publish("person/duration", json.dumps({"duration": duration}))

            # Send frame to the ffmpeg server
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

            # Write out the frame
            if image_flag:
                cv2.imwrite('output_image.jpg', frame)
            else:
                # cv2.putText(frame, f"{current_cout} | {total_count}", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness = 1)
                out.write(frame)

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
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
