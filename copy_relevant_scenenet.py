import argparse
import os
import shutil

import numpy as np

from scenenet_loader import find_trajectories, ms_per_frame, load_depth_map_in_mm
from joblib import Parallel, delayed


parser = argparse.ArgumentParser(description='')
parser.add_argument('traj_dir')
parser.add_argument('to_dir')


def filter_scenenet(trajectories, trajectory_indices):
    paths_and_frames = []
    for path in trajectories:
        for i in trajectory_indices:
            depth = load_depth_map_in_mm(os.path.join(path, "depth", "%d.png" % (i * ms_per_frame)))
            if np.any(depth):
                paths_and_frames.append((path, i))

    print("Found %d paths_and_frames" % len(paths_and_frames))

    return paths_and_frames


if __name__ == "__main__":
    args = parser.parse_args()
    from_path = args.traj_dir
    to_path = args.to_dir

    # from_path, to_path should both end with '/'
    if from_path[-1] != '/' or to_path[-1] != '/':
        raise(RuntimeError("Args have to be directories with trailing slash '/'"))

    print(from_path)
    print(to_path)

    traj_indices = range(0, 300, 13)  # total of 24 per trajectory

    trajectories = find_trajectories(from_path)
    print("Got Trajectories")

    batch_size = 128
    traj_partition = [trajectories[i:i+batch_size] for i in range(0, len(trajectories), batch_size)]
    #traj_partition = [trajectories[i:i+batch_size] for i in range(0, batch_size*3, batch_size)]

    filtered = Parallel(n_jobs=6)(delayed(filter_scenenet)(ts, traj_indices) for ts in traj_partition)

    paths_to_copy = [p for ps in filtered for p in ps]

    for path, i in paths_to_copy:
        if not path.startswith(from_path):
            raise(RuntimeError("Path has wrong prefix!"))
        photo_from_path = os.path.join(path, "photo", "%d.jpg" % (i * ms_per_frame))
        depth_from_path = os.path.join(path, "depth", "%d.png" % (i * ms_per_frame))
        rel_path = path[len(from_path):]
        photo_dir_to_path = os.path.join(to_path, rel_path, "photo")
        depth_dir_to_path = os.path.join(to_path, rel_path, "depth")
        photo_to_path = os.path.join(photo_dir_to_path, "%d.jpg" % (i * ms_per_frame))
        depth_to_path = os.path.join(depth_dir_to_path, "%d.png" % (i * ms_per_frame))
        if not os.path.isdir(photo_dir_to_path):
            print("mkdir %s" % photo_dir_to_path)
            os.makedirs(photo_dir_to_path)
        if not os.path.isdir(depth_dir_to_path):
            print("mkdir %s" % depth_dir_to_path)
            os.makedirs(depth_dir_to_path)
        if not os.path.isfile(photo_to_path):
            print("Copy from %s to %s" % (photo_from_path, photo_to_path))
            shutil.copyfile(photo_from_path, photo_to_path)
        if not os.path.isfile(depth_to_path):
            print("Copy from %s to %s" % (depth_from_path, depth_to_path))
            shutil.copyfile(depth_from_path, depth_to_path)


