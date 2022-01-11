import shutil
import time
import math
import yolo
import trace
import utils
import os
import csv

# Lifetime for a target, detected targets will be sent to server within its lifetime
T = 10000
# Confidence threshold
C = 0.8
# Dataset path
img_path = os.path.join("dataset/trafficcam_2/src")


def main():
    global C
    for direc in ['./cache', './temp']:
        if os.path.exists(direc):
            shutil.rmtree(direc)
        os.mkdir(direc)
    for file in ['./result_bw.csv']:
        if os.path.isfile(file):
            os.remove(file)
        f = open(file, 'w+')
        f.close()
    last_frame = []
    # Origin Dateset
    raw_set = os.listdir('./dataset/trafficcam_2/src')
    raw_set.sort()
    # Sampling Function
    # raw_set = utils.generate_data()
    for raw_img in raw_set:
        bw = 0
        raw_path = os.path.join(img_path, raw_img)
        raw_current = yolo.detect(raw_path)
        current_frame = []
        if len(last_frame) > 0:
            print(
                '\n-----------------------------------------------------------------------------------------\n')
            for lt in last_frame:
                print(lt)
        print('\nCURRENT TARGETS：')
        for index, target in enumerate(raw_current):
            t = {'name': str(index) + '_' + raw_img, 'shape': target[:4], 'confidence': target[4],
                 'result': target[5],
                 'detected': False,
                 'birth': math.floor(time.time() * 1000 % T)}
            current_frame.append(t)
            print('target', index, ':', t)
        last_frame, current_frame, tracked = trace.preprocess_data(last_frame, current_frame)
        print('\nTracked.\n')
        put_back = []
        if len(tracked) > 0:
            print('Found same targets:', '\n', tracked)
            for pair in tracked:
                last_target = utils.find_target(last_frame, pair[0])
                temp_target = utils.find_target(current_frame, pair[1])
                if not last_target['detected']:
                    if float(temp_target['confidence']) >= C:
                        print('Because of enough CONFIDENCE,')
                        utils.cache_append(temp_target)
                        utils.cache_pop(last_target['name'])
                        # Send targets which are slightly upon C to server entirely to adjust confidence threshold
                        if float(temp_target['confidence']) - C < 0.05:
                            bw, nC = utils.send_image_to_server(temp_target, utils.check_age(last_target), bw)
                            if not nC == 0.0:
                                C = nC
                        else:
                            utils.send_to_server(temp_target)
                        temp_target['detected'] = True
                        put_back.append(temp_target)
                    else:
                        if float(temp_target['confidence']) > float(last_target['confidence']):
                            print('Because of higher confidence,')
                            temp_target['birth'] = last_target['birth']
                            utils.cache_append(temp_target)
                            utils.cache_pop(last_target['name'])
                            if utils.check_age(last_target) > utils.AGE:
                                print('Because of time limit,')
                                bw, nC = utils.send_image_to_server(temp_target, utils.check_age(last_target), bw)
                                if not nC == 0.0:
                                    C = nC
                                temp_target['detected'] = True
                            put_back.append(temp_target)
                        else:
                            print('Because of lower confidence,')
                            bw, nC = utils.send_image_to_server(last_target, utils.check_age(last_target), bw)
                            if not nC == 0.0:
                                C = nC
                            last_target['detected'] = True
                            put_back.append(last_target)
                else:
                    print('Save detected target ', temp_target['name'], ' .')
                    temp_target['detected'] = True
                    utils.cache_append(temp_target)
                    utils.cache_pop(last_target['name'])
                    put_back.append(temp_target)
        for lost_target in last_frame:
            print('Because of target lost,')
            if not lost_target['detected']:
                bw, nC = utils.send_image_to_server(lost_target, utils.check_age(lost_target), bw)
                if not nC == 0.0:
                    C = nC
            utils.cache_pop(lost_target['name'])
        for index, new_found in enumerate(current_frame):
            print('New target detected:\n', new_found)
            utils.cache_append(new_found)
            if float(new_found['confidence']) >= C:
                print('Because of enough CONFIDENCE,')
                utils.send_to_server(new_found)
                new_found['detected'] = True
        current_frame.extend(put_back)
        # Send all cached targets before ending the process
        # if raw_img == raw_set[-1]:
        #     for target in current_frame:
        #         if not target['detected']:
        #             print("Because it's the last frame,")
        #             bw, nC = utils.send_image_to_server(target, utils.check_age(last_target), bw)
        # else:
        last_frame = current_frame
        # write bw
        results_files = open("result_bw.csv", "a")
        csv_writer = csv.writer(results_files)
        csv_writer.writerow([bw])
        results_files.close()
        print(bw / 1024)


if __name__ == '__main__':
    main()
