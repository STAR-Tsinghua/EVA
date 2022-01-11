import os
import logging
from .object_detector import Detector
from utils import (Results, Region)
import cv2 as cv


class Server:
    def __init__(self):
        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.detector = Detector()
        self.logger.info("Server started")

    def perform_detection(self, images_direc, resolution, fname=None):
        final_results = Results()
        rpn_regions = Results()
        # read image
        fid = int(fname.split(".")[0])
        image_path = os.path.join(images_direc, fname)
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # detect image
        detection_results, rpn_results = (
            self.detector.infer(image))
        frame_with_no_results = True
        for label, conf, (x, y, w, h) in detection_results:
            if w * h == 0.0:
                continue
            r = Region(fid, x, y, w, h, conf, label,
                       resolution)
            final_results.append(r)
        for label, conf, (x, y, w, h) in rpn_results:
            r = Region(fid, x, y, w, h, conf, label,
                       resolution)
            rpn_regions.append(r)
        self.logger.debug(
            f"Got {len(final_results)} results "
            f"and {len(rpn_regions)} for {fname}")

        return final_results, rpn_regions

