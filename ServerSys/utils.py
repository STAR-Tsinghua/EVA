import csv
import re
import numpy as np
import networkx
from networkx.algorithms.components.connected import connected_components


def calc_intersection_area(a, b):
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.h, b.y + b.h)
    ri = min(a.x + a.w, b.x + b.w)

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a.w)
    h = max(0, a.h)

    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def filter_bbox_group(bb1, bb2, iou_threshold):
    if calc_iou(bb1, bb2) > iou_threshold and bb1.label == bb2.label:
        return True
    else:
        return False


def pairwise_overlap_indexing_list(single_result_frame, iou_threshold):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i, j] for i, x in enumerate(single_result_frame)
                for j, y in enumerate(single_result_frame)
                if i != j if filter_bbox_group(x, y, iou_threshold)]
    return pointwise + pairwise


def simple_merge(single_result_frame, index_to_merge):
    # directly using the largest box
    bbox_large = []
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.x + x.w)
        bottom = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.y + x.h)

        fid, x, y, w, h, conf, label, resolution = (
            left.fid, left.x, top.y, right.x + right.w - left.x,
            bottom.y + bottom.h - top.y, left.conf, left.label,
            left.resolution)
        single_merged_region = Region(fid, x, y, w, h, conf,
                                      label, resolution)
        bbox_large.append(single_merged_region)
    return bbox_large


def merge_boxes_in_results(results_dict, min_conf_threshold, iou_threshold):
    final_results = Results()

    # Clean dict to remove min_conf_threshold
    for _, regions in results_dict.items():
        to_remove = []
        for r in regions:
            if r.conf < min_conf_threshold:
                to_remove.append(r)
        for r in to_remove:
            regions.remove(r)

    for fid, regions in results_dict.items():
        overlap_pairwise_list = pairwise_overlap_indexing_list(
            regions, iou_threshold)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(
            connected_components(overlap_graph), key=len, reverse=True)]
        merged_regions = simple_merge(regions, grouped_bbox_idx)
        for r in merged_regions:
            final_results.append(r)
    return final_results


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution):
        self.fid = int(fid)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.label = label
        self.resolution = float(resolution)

    @staticmethod
    def convert_from_server_response(r, res):
        return Region(r[0], r[1], r[2], r[3], r[4], r[5], r[6], res)

    def __str__(self):
        string_rep = (f"{self.fid}, {self.x:0.3f}, {self.y:0.3f}, "
                      f"{self.w:0.3f}, {self.h:0.3f}, {self.conf:0.3f}, "
                      f"{self.label}")
        return string_rep

    def is_same(self, region_to_check, threshold=0.5):
        # If the fids or labels are different
        # then not the same
        if (self.fid != region_to_check.fid or
                ((self.label != "-1" and region_to_check.label != "-1") and
                 (self.label != region_to_check.label))):
            return False

        # If the intersection to union area
        # ratio is greater than the threshold
        # then the regions are the same
        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False

    def enlarge(self, ratio):
        x_min = max(self.x - self.w * ratio, 0.0)
        y_min = max(self.y - self.h * ratio, 0.0)
        x_max = min(self.x + self.w * (1 + ratio), 1.0)
        y_max = min(self.y + self.h * (1 + ratio), 1.0)
        self.x = x_min
        self.y = y_min
        self.w = x_max - x_min
        self.h = y_max - y_min

    def copy(self):
        return Region(self.fid, self.x, self.y, self.w, self.h, self.conf,
                      self.label, self.resolution)


class Results:
    def __init__(self):
        self.regions = []
        self.regions_dict = {}

    def __len__(self):
        return len(self.regions)

    def results_high_len(self, threshold):
        count = 0
        for r in self.regions:
            if r.conf > threshold:
                count += 1
        return count

    def is_dup(self, result_to_add, threshold=0.5):
        # return the regions with IOU greater than threshold
        # and maximum confidence
        if result_to_add.fid not in self.regions_dict:
            return None

        max_conf = -1
        max_conf_result = None
        for existing_result in self.regions_dict[result_to_add.fid]:
            if existing_result.is_same(result_to_add, threshold):
                if existing_result.conf > max_conf:
                    max_conf = existing_result.conf
                    max_conf_result = existing_result
        return max_conf_result

    def combine_results(self, additional_results, threshold=0.5):
        for result_to_add in additional_results.regions:
            self.add_single_result(result_to_add, threshold)

    def add_single_result(self, region_to_add, threshold=0.5):
        if threshold == 1:
            self.append(region_to_add)
            return
        dup_region = self.is_dup(region_to_add, threshold)
        if (not dup_region or
                ("tracking" in region_to_add.origin and
                 "tracking" in dup_region.origin)):
            self.regions.append(region_to_add)
            if region_to_add.fid not in self.regions_dict:
                self.regions_dict[region_to_add.fid] = []
            self.regions_dict[region_to_add.fid].append(region_to_add)
        else:
            final_object = None
            if dup_region.origin == region_to_add.origin:
                final_object = max([region_to_add, dup_region],
                                   key=lambda r: r.conf)
            elif ("low" in dup_region.origin and
                  "high" in region_to_add.origin):
                final_object = region_to_add
            elif ("high" in dup_region.origin and
                  "low" in region_to_add.origin):
                final_object = dup_region
            dup_region.x = final_object.x
            dup_region.y = final_object.y
            dup_region.w = final_object.w
            dup_region.h = final_object.h
            dup_region.conf = final_object.conf
            dup_region.origin = final_object.origin

    def suppress(self, threshold=0.5):
        new_regions_list = []
        while len(self.regions) > 0:
            max_conf_obj = max(self.regions, key=lambda e: e.conf)
            new_regions_list.append(max_conf_obj)
            self.remove(max_conf_obj)
            objs_to_remove = []
            for r in self.regions:
                if r.fid != max_conf_obj.fid:
                    continue
                if calc_iou(r, max_conf_obj) > threshold:
                    objs_to_remove.append(r)
            for r in objs_to_remove:
                self.remove(r)
        new_regions_list.sort(key=lambda e: e.fid)
        for r in new_regions_list:
            self.append(r)

    def append(self, region_to_add):
        self.regions.append(region_to_add)
        if region_to_add.fid not in self.regions_dict:
            self.regions_dict[region_to_add.fid] = []
        self.regions_dict[region_to_add.fid].append(region_to_add)

    def remove(self, region_to_remove):
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)
        self.regions.remove(region_to_remove)
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)

    def fill_gaps(self, number_of_frames):
        if len(self.regions) == 0:
            return
        results_to_add = Results()
        max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        for i in range(number_of_frames):
            if i not in fids_in_results:
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     0.1, "no obj",
                                                     max_resolution))
        self.combine_results(results_to_add)
        self.regions.sort(key=lambda r: r.fid)

    def write_results_txt(self, fname):
        results_file = open(fname, "w")
        for region in self.regions:
            # prepare the string to write
            str_to_write = (f"{region.fid},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf}\n")
            results_file.write(str_to_write)
        results_file.close()

    def write_results_csv(self, fname):
        results_files = open(fname, "w")
        csv_writer = csv.writer(results_files)
        for region in self.regions:
            row = [region.fid, region.x, region.y,
                   region.w, region.h,
                   region.label, region.conf,
                   region.resolution, region.origin]
            csv_writer.writerow(row)
        results_files.close()

    def write(self, fname):
        if re.match(r"\w+[.]csv\Z", fname):
            self.write_results_csv(fname)
        else:
            self.write_results_txt(fname)