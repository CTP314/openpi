import shutil
import json
from pathlib import Path

import actmem_bench.envs
import gymnasium as gym
import numpy as np

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import h5py
import tyro
import math
import mani_skill

REPO_NAME = "actmem_bench"

def main(h5_path: Path, *, push_to_hub: bool = False):
    json_path = h5_path.with_suffix('.json')
    assert json_path.exists(), f"JSON file not found: {json_path}"
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    assert 'env_info' in json_data, "JSON file does not contain 'env_info'"
    
    h5_file = h5py.File(h5_path, 'r')
    
    image_shape = h5_file['traj_0']['obs']['sensor_data']['base_camera']['rgb'].shape[1:]
    wrist_image_shape = h5_file['traj_0']['obs']['sensor_data']['hand_camera']['rgb'].shape[1:]
    
    # arm_state_shape = h5_file['traj_0']['obs']['extra']['tcp_pose'].shape[1:]
    # gripper_state_shape = (1,)
    # state_shape = (arm_state_shape[0] + gripper_state_shape[0],)
    
    state_shape = h5_file['traj_0']['obs']['agent']['qpos'].shape[1:]
    
    action_shape = h5_file['traj_0']['actions'].shape[1:]
    
    control_mode = json_data['env_info']['env_kwargs']['control_mode']
    obs_mode = json_data['env_info']['env_kwargs']['obs_mode']
    shader_type = json_data['env_info']['env_kwargs']['sensor_configs']['shader_pack']
    env_id = json_data['env_info']['env_id']
    
    env = gym.make(
        env_id,
        **json_data['env_info']['env_kwargs'],
    )
    
    repo_name = f"{REPO_NAME}-{env_id}-{control_mode}-{obs_mode}-{shader_type}"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        confirm = input(f"Output path {output_path} already exists. Do you want to delete it? (y/n): ")
        if confirm.lower() == 'y':
            print(f"Deleting {output_path}...")
            shutil.rmtree(output_path)
        elif confirm.lower() == 'n':
            print("Exiting without deleting the existing dataset.")
            return

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=env.unwrapped._control_freq,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    for traj_idx in range(len(h5_file.keys())):
        traj = h5_file[f'traj_{traj_idx}']
        obs = traj['obs']
        actions = traj['actions']
        
        for step in range(actions.shape[0]):
            image = obs['sensor_data']['base_camera']['rgb'][step]
            wrist_image = obs['sensor_data']['hand_camera']['rgb'][step]
            gripper_state = obs['agent']['qpos'][step][..., -2:]
            action = actions[step]
            state = np.concatenate([
                # obs['extra']['tcp_pose'][step][:3],
                # _quat2euler(obs['extra']['tcp_pose'][step][3:]),
                obs['agent']['qpos'][step][..., :-2],  # Exclude gripper state
                gripper_state
            ], axis=-1)
            
            dataset.add_frame({
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": action,   
            }, task=env_id)      
            
        dataset.save_episode()  
    
    if push_to_hub:
        dataset.push_to_hub(
            tags=["actmem_bench", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        
def _quat2euler(quat):
    """
    w, x, y, z -> euler x y z
    """
    assert quat.shape[-1] == 4, "Quaternion must have shape (..., 4)"
    w, x, y, z = np.split(quat, 4, axis=-1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.concatenate([roll_x, pitch_y, yaw_z], axis=-1)


if __name__ == "__main__":
    tyro.cli(main)