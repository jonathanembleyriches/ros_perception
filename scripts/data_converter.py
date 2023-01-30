from ouster import client, pcap
import numpy as np
import os
import argparse
from contextlib import closing
from typing import Tuple, List
import matplotlib.pyplot as plt  # type: ignore
from more_itertools import nth
import numpy as np
from sklearn.neighbors import KDTree
from utils.data_process import DataProcessing as DP
import os
import yaml
import pickle

from os.path import join, exists


def load_pcap_and_meta(pcap_path: str, meta_path: str):

    with open(meta_path, "r") as f:
        metadata = client.SensorInfo(f.read())
    source = pcap.Pcap(pcap_path, metadata)
    return source, metadata


def extract_scan_at_index(
    source: client.PacketSource, metadata: client.SensorInfo, num: int
) -> int:
    scan = nth(client.Scans(source), num)
    if not scan:
        print(f"ERROR: Scan # {num} in not present in pcap file")
        exit(1)
    # transform data to 3d points and graph
    xyzlut = client.XYZLut(metadata)
    xyz = xyzlut(scan.field(client.ChanField.RANGE))

    [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
    pc = np.column_stack((x, y, z)).astype(np.float32)
    print(pc.dtype)
    return pc


def folder_file_manager(out_dir,exp_name, seq):
    full_out_dir = os.path.abspath(out_dir)
    exp_path = full_out_dir + "/" +exp_name 
    kd_path = exp_path+ "/" + "KDTree"

    proj_path = exp_path+ "/" + "proj"
    velo_path = exp_path+ "/" + "velodyne"


    os.makedirs(exp_name) if not exists(exp_name) else None
    # os.makedirs(seq_path) if not exists(seq_path) else None
    os.makedirs(kd_path) if not exists(kd_path) else None
    os.makedirs(proj_path) if not exists(proj_path) else None
    os.makedirs(velo_path) if not exists(velo_path) else None
    return kd_path, proj_path, velo_path

def process_for_randlanet(
    points,exp_name, index, save_intermediate=False, out_dir="intermediate_data"
):

    points = points.reshape((-1, 4))
    points = points[:, 0:3]  # get xyz
    sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
    search_tree = KDTree(sub_points)
    proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
    proj_inds = proj_inds.astype(np.int32)
    if save_intermediate:
        kd_path, proj_path, velo_path = folder_file_manager(out_dir, exp_name, index)
        KDTree_save = f"{kd_path}/{index}.pkl"
        proj_save = f"{proj_path}/{index}_proj.pkl"

        np.save(f"{velo_path}/{index}", sub_points)
        with open(KDTree_save, "wb") as f:
            pickle.dump(search_tree, f)
        with open(proj_save, "wb") as f:
            pickle.dump([proj_inds], f)





source_path = "/home/jon/Downloads/OS-1-128_v2.3.0_1024x10_20220419_161749-000.pcap"
meta_path = "/home/jon/Downloads/OS-1-128_v2.3.0_1024x10_20220419_161749.json"
source, metadata = load_pcap_and_meta(source_path, meta_path)
pc = extract_scan_at_index(source, metadata, 0)
process_for_randlanet(pc,"00",0,save_intermediate=True)
print(pc)
