import collections
import dataclasses
import math
import pathlib
import time
import sys

import imageio
import gymnasium as gym
import actmem_bench.envs
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from loguru import logger  
import tqdm
import tyro
import pandas as pd

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10

    #################################################################################################################
    # actmem_bench environment-specific parameters
    #################################################################################################################
    task: str = "PickCylinderByMemory-v1"
    task_suites: list[str] = dataclasses.field(
        default_factory=lambda: [
            'ind', 'ind_interpolation', 'ood_interpolation', 'ood_extrapolation', 'ood_distractor'
        ]
    )
    control_mode: str = "pd_joint_pos"
    obs_mode: str = "rgbd"
    num_trials: int = 100
    sensor_configs: dict = dataclasses.field(default_factory=lambda: {
        'height': 256,
        'width': 256,
        'shader_pack': 'rt',
    })

    #################################################################################################################
    # Utils
    #################################################################################################################
    # video_out_path: str = "data/libero/videos"  # Path to save videos
    result_out_path: pathlib.Path = pathlib.Path("data/actmem_bench")

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    env = gym.make(args.task, control_mode=args.control_mode, obs_mode=args.obs_mode, sensor_configs=args.sensor_configs, sim_backend="cpu")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in args.task_suites:
        task_description = task_id
        
        results = []
        date = time.strftime("%Y-%m-%d_%H-%M-%S")
        result_out_path = args.result_out_path / args.task / task_id / date
        result_out_path.mkdir(parents=True, exist_ok=True)
        run_cmd = " ".join(sys.argv)
        (result_out_path / "run_cmd.txt").write_text(run_cmd)
        video_out_path = result_out_path / "videos"
        video_out_path.mkdir(parents=True, exist_ok=True)
        traj_out_path = result_out_path / "trajs"
        traj_out_path.mkdir(parents=True, exist_ok=True)
        
        options = env.unwrapped.generate_options(task_id)
        if options is not None:
            num_trials = len(options)
        else:
            num_trials = args.num_trials
            options = [None] * num_trials
        
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(num_trials), ncols=0):
            obs, info = env.reset(options=options[episode_idx])
            action_plan = collections.deque()
            
            t = 0
            replay_images = []
            
            st_time = time.time()
            done = False
            success = False
            
            action_traj = []
            
            while not done:
                try:
                    img = obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy()[0]
                    wrist_img = obs["sensor_data"]["hand_camera"]["rgb"].cpu().numpy()[0]
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    
                    replay_images.append(img)
                    
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    # obs["extra"]["tcp_pose"].cpu().numpy()[0, :3],
                                    # _quat2euler(obs["extra"]["tcp_pose"].cpu().numpy()[0, 3:]),
                                    obs["agent"]["qpos"][..., :-2].cpu().numpy()[0],  # Exclude gripper state
                                    obs["agent"]["qpos"][..., -2:].cpu().numpy()[0] 
                                )
                            ),
                            "prompt": str(args.task),
                        }
                        
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[:args.replan_steps])
                        
                    action = action_plan.popleft()

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    action_traj.append(action)
                    
                    if done:
                        if info.get("success", False):
                            success = True
                            task_successes += 1
                            total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logger.error(f"Caught exception: {e}")
                    break
                
            task_episodes += 1
            total_episodes += 1
            
            # Save a replay video of the episode
            suffix = success and "success" or "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                video_out_path / f"rollout_{task_segment}_{suffix}_{episode_idx}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            np.savez((traj_out_path / f"{episode_idx}").as_posix(), action=action_traj)
            result = dict(
                task_id=task_id,
                task_description=task_description,
                success=success,
                eval_time=time.time() - st_time
            )
            result.update(options[episode_idx] if options[episode_idx] is not None else {})
            results.append(result)
            # Save results to CSV
            df = pd.DataFrame(results)
            df.to_csv(result_out_path / "results.csv", index=False)

            # Log current results
            logger.info(f"Success: {success}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

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
    logger.info("Starting evaluation...")
    tyro.cli(eval_libero)
