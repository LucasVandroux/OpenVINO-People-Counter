#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

from utils import BoundingBox
import numpy as np


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_xml_path, device="CPU"):
        """ Load the model 

            Args:
                model_xml_path (str): path to the .xml files representing the model. It is assumed the .bin file with the weights has the same name.
                device (str: 'CPU'): name of the device where to load the model
        """
        # Create path to both files of the model
        model_xml = model_xml_path
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        # Initialize the plugin
        self.plugin = IECore()

        # Read IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Check for supported layers
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            sys.exit(f"Unsupported layers found: {unsupported_layers}.")

        # No need of CPU extension with the 2020.3 version of OpenVINO
        # plugin.add_extension(extension_path=cpu_ext, device_name="CPU")

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer and output layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def get_input_shape(self):
        """ Gets the input shape of the network
        """
        return self.network.inputs[self.input_blob].shape

    def async_inference(self, image, request_id=0):):
        """ Makes an asynchronous inference request, given an input image.

        Args:
            image (np.array): numpy array representing the image
            request_id (int: 0): request id used to identify the request
        """
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})

    def wait(self):
        """ Checks the status of the inference request.
        """
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        """ Returns a list of the results for the output layer of the network.
        """
        return self.exec_network.requests[0].outputs[self.output_blob]

    def postprocess_output(self, network_output, width, height, prob_threshold = 0.5):
        """ Post-process the output of the network

        Args:
            network_output: direct output of the network after inference
            width (int): width of the input frame
            height (int): height of the input frame
            prob_threshold (float: 0.5): probability threshold for detections filtering

        Returns:
            list_detections (list[(x: int, y:int, w:int, h:int)]): list of bounding boxes of the 
                detected objects. A bounding boxe is a tuple (x, y, w, h) where (x,y) are the   
                coordinates of the top-left corner of the bounding box and w its width and h its 
                height in pixels.
        """
        list_detections = []
        # The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected 
        # bounding boxes. For each detection, the description has the format: 
        # [image_id, label, conf, x_min, y_min, x_max, y_max]
        bboxes = np.reshape(network_output, (-1, 7)).tolist()

        for bbox in bboxes:
            conf = bbox[2]
            object_class = int(bbox[1]) # Using VOC labels --> 15 == person
            if conf >= prob_threshold and object_class == 15:
                xmin = int(bbox[3] * width)
                ymin = int(bbox[4] * height)
                xmax = int(bbox[5] * width)
                ymax = int(bbox[6] * height)

                list_detections.append(BoundingBox(xmin, ymin, (xmax - xmin), (ymax - ymin)))

        return list_detections


