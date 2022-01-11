import math
import os
import random
import time
import requests
import cv2 as cv
import json

HNAME = '127.0.0.1:5001'
DATAPATH = './dataset/trafficcam_2/src/'

T = 10000
# Assumed transmission delay
L = 500
# Maximum bit rate
R = 10000
# frequency times bandwidth
fB = 2
# Max survive time for image could send by max R
AGE = T - L - R / fB


# Send a confident recognized result to server
def send_to_server(target):
    to_send = {'name': target['name'], 'shape': target['shape'], 'conf': target['confidence'],
               'label': target['result']}
    response = requests.Session().post(
        "http://" + HNAME + "/high", data=json.dumps(to_send))
    # response_json = json.loads(response.text)
    print('Successfully sent result ', target['name'], ' to server!')


# Send pixels of a not confident enough target to server
# Attribute 'age' belongs to the target's ancestor not its own birth
def send_image_to_server(target, age, bw):
    # ABR algorithms can be inserted
    r = (T - age - L) * fB
    r = r if r < R else R
    # compress image
    raw_path = './cache/' + target['name']
    raw = cv.imread(raw_path)
    temp_path = './temp/' + target['name']
    cv.imwrite(temp_path, raw, [cv.IMWRITE_PNG_COMPRESSION, 9])
    to_send_file = {'image': open(temp_path, 'rb')}
    bw += os.path.getsize(temp_path)
    print("Accumulated Bandwidth: ", bw)
    to_send_data = {"name": target['name'], "shape": target['shape'], 'conf': target['confidence'],
                    'label': target['result'], 'bw': os.path.getsize(temp_path)}
    response = requests.Session().post(
        "http://" + HNAME + "/low", files=to_send_file, data=to_send_data)
    nC = float(response.content)
    print('Successfully sent image ', target, ' to server with r as ' + str(r) + ' (max=10000).')
    return bw, nC


# Cache a template target image. Works when new targets comes and save last frame's targets
def cache_append(img):
    position = img['shape']
    raw_path = DATAPATH + img['name'][-14:]
    raw = cv.imread(raw_path)
    segment = './cache/' + img['name']
    a = raw[position[0]:position[2], position[1]:position[3]]
    cv.imwrite(segment, a, [cv.IMWRITE_PNG_COMPRESSION, 0])
    print('Picture ', img['name'], ' appended into cache!')


# Pop target images that disappear from scene or decline in confidence
def cache_pop(img):
    files = os.listdir("./cache")
    img_path = './cache/' + img
    os.remove(img_path)
    print('Picture', img, ' popped from cache!')


# Return the queried target's survival time
def check_age(target):
    age = math.floor(time.time() * 1000 % T) - int(target['birth'])
    age = age if age > 0 else age + T
    return age


# From targets find target with name 'name'
def find_target(targets, name):
    for index, target in enumerate(targets):
        if target['name'] == name:
            print("Found target: ", target)
            return targets.pop(index)
    return -1


# Sampling function
def generate_data():
    raw = os.listdir("./dataset/trafficcam_2/src")
    raw.sort()
    start = random.randint(0, 30)
    raw = raw[start:start + 30]
    rm = []
    for i in range(15):
        rm.append(i * 2)
    gen = [i for num, i in enumerate(raw) if num not in rm]
    return gen
