import math

# Trace threshold
THRESHOLD = 0.6


def calc_intersection_area(a, b):
    to = max(a[1], b[1])
    le = max(a[0], b[0])
    bo = min(a[3], b[3])
    ri = min(a[2], b[2])

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a[2] - a[0])
    h = max(0, a[3] - a[1])

    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area


# Return target pairs in two adjacent frames that detected to present same objects
def preprocess_data(last_frame, current_frame):
    res = []
    temp = []
    to_remove = []
    for index, target in enumerate(current_frame):
        for l_index, l_target in enumerate(current_frame):
            if index is not l_index:
                a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
                b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
                iou = calc_iou(a, b)
                if iou > THRESHOLD and target['name'].split(".")[0].split("_")[1] == \
                        l_target['name'].split(".")[0].split("_")[1]:
                    # print('dump: ', l_target['name'], target['name'])
                    if target['confidence'] > l_target['confidence']:
                        to_remove.append(l_target)
                    else:
                        to_remove.append(target)
    # print(to_remove)
    for item in to_remove:
        if item in current_frame:
            current_frame.remove(item)

    if len(last_frame) > 0:
        for index, target in enumerate(current_frame):
            max_iou = 0
            temp_pair = []
            remove_index = 0
            remove_target = []
            a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
            for l_index, l_target in enumerate(last_frame):
                b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
                iou = calc_iou(a, b)
                if iou > THRESHOLD:
                    if iou > max_iou:
                        temp_pair = [l_target['name'], target['name']]
                        max_iou = iou
                        remove_index = l_index
                        remove_target = l_target
            # print(target['name'], remove_target['name'], max_iou)
            if temp_pair:
                # print(temp_pair)
                res.append(temp_pair)
                last_frame.pop(remove_index)
                temp.append(remove_target)
    last_frame.extend(temp)
    return last_frame, current_frame, res
