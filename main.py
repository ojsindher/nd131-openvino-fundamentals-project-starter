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
from datetime import datetime

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


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
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so",
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
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, width, height, prob_threshold):
    count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold and box[1] == 1:
            count+=1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 3)
    return frame, count

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
    model = args.model
    d = args.device
    c = args.cpu_extension
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, d, c)
    (n,c,h,w) = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    
    image_flag = False
    if args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
    #Remove it later
    if not image_flag:
        out = cv2.VideoWriter('out.mp4', 0x00000021, fps, (width, height))
    else:
        out = None
    ### TODO: Loop until stream is over ###
    total_ = 0
    count_flag_2 = 0
    time_diff = np.array([])
    int_t = np.array([])
    count_a = np.zeros(32)
    duration_avg = 0
    duration_av = 0
    i = 32
    pif = 0
    actual_pif = 1050
    duration = 0
    current_count = 0
    total_count = 0
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        ret, frame = cap.read()

        if not ret:
            break
        i+=1
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (w,h)).transpose(2,0,1)
        p_frame = p_frame.reshape(1, c, h, w)
        ### TODO: Start asynchronous inference for specified request ###
        int_t1 = datetime.now()
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait()==0:
            ### TODO: Get the results of the inference request ###
            int_t2 = datetime.now()
            inf_tt = 1000*((int_t2 - int_t1).total_seconds())
            int_t = np.append(int_t, inf_tt)
            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            pp_frame, count = draw_boxes(frame, result, width, height, prob_threshold)
                ### TODO: Calculate and send relevant information on ###
                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###
            if count>=1:
                pif+=1
                
            count_a = np.append(count_a, count)
            count_flag_1 = 0
            if np.mean(count_a[i-32:i-1]) > 0.65:
                count_flag_1 = 1
                client.publish("person", json.dumps({"count": count}))
            
            if count >= 1 and np.mean(count_a[i-20:i-1]) > 0.7:
                client.publish("person", json.dumps({"count": count}))
            elif count == 0 and np.mean(count_a[i-20:i-1]) < 0.3:
                client.publish("person", json.dumps({"count": count}))
            
            if count >= 1 and count_flag_1 == 1 and count_flag_2 == 0:
                person_appears = time.time()
                total_+=count
                count_flag_2 = count_flag_1
            elif np.mean(count_a[i-15:i-1]) < 0.5 and count_flag_1 == 0 and count_flag_2 == 1:
                person_leaves = time.time()
                diff = int(person_leaves - person_appears)
                duration_av = round(diff, 2)
                time_diff = np.append(time_diff, diff)
                duration_avg = round(np.mean(time_diff), 2)
                count_flag_2 = count_flag_1
                client.publish("person/duration", json.dumps({"duration": duration_av}))
            
            cv2.putText(pp_frame, "Count: {}, Total: {}, Time spent avg: {} sec, Inference time avg.: {} ms".format(count, total_, duration_avg, round(np.mean(int_t), 3)), (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,10,10), 1)
            
        
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(pp_frame)
        sys.stdout.flush()
            
            # Break if escape key pressed
        if key_pressed == 27:
            break

        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('out_img.jpg', pp_frame)   
    if not image_flag:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

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
