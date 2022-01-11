"""
predict low confidence objects
"""
import os
from backend.server import Server
from utils import merge_boxes_in_results
server = Server()
image_direc = f"./server_temp/"
f = open("./backend/low_img.txt", "r")
results_file = open("./backend/high_img.txt", "a")
for line in f:
    name = line.split(",")[0]
    img_path = f"./server_temp/{name}"
    if os.path.exists(img_path):
        y1 = line.split(",")[1]
        x1 = line.split(",")[2]
        y2 = line.split(",")[3]
        x2 = line.split(",")[4]
        conf = line.split(",")[5]
        label = line.split(",")[6]
        #if float(conf) < 0.8:
        results, rpn_results = server.perform_detection(image_direc, 1.0, name)
        results = merge_boxes_in_results(results.regions_dict, 0.3, 0.3)
        for region in results.regions:
            # prepare the string to write
            conf = region.conf
            label = region.label
            str_to_write = f"{name}, {y1}, {x1}, {y2}, {x2}, {conf}, {label}\n"
            results_file.write(str_to_write)
        # else:
        #     results_file.write(line)
f.close()
results_file.close()