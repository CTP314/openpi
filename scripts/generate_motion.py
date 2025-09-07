import dataclasses
import enum
import logging
import socket
import pathlib
import tqdm

import tyro
import torch
import numpy as np

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.training.data_loader as _data_loader

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint = dataclasses.field(default_factory=Checkpoint)

    # Directory to save output files
    output_dir: pathlib.Path = pathlib.Path("output/tmp")
    
    # Num samples to generate
    num_samples: int = 16
    
    # Num steps to fix
    num_fixed_prefix: int = 0


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
    )
    
def main(args: Args) -> None:
    config = _config.get_config(args.policy.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(
        data_config=data_config, action_horizon=config.model.action_horizon, model_config=config.model, drop_last=False
    )
    transformed_dataset = _data_loader.transform_dataset(dataset, data_config)
    unwrapped_dataset = dataset._dataset
    state = np.zeros_like(dataset[0]['state'].numpy())
    states = np.array([state] * config.model.action_horizon)
    actions = np.zeros_like(transformed_dataset[0]['actions'])
    image = np.zeros_like(dataset[0]['image'].numpy())
    wrist_image = np.zeros_like(dataset[0]['wrist_image'].numpy())
    
    if args.default_prompt:
        prompt = args.default_prompt
    else:
        prompt =  "pick up the black bowl between the plate and the ramekin and place it on the plate"
    ep_idx = -1
    ep_len = -1
    for k, v in unwrapped_dataset.meta.episodes.items():
        if prompt in v['tasks']:
            ep_idx = k
            ep_len = v['length']
            break
    if ep_idx != -1:
        i = unwrapped_dataset.episode_data_index['from'][ep_idx].item()
        logging.info(f"Found matching prompt in dataset at index {i}. Using corresponding state and images.")
        data = dataset[i]
        state = data['state'].numpy()
        image = data['image'].numpy()
        wrist_image = data['wrist_image'].numpy()
        actions = transformed_dataset[i]['actions']
        states = []
        for j in tqdm.trange(config.model.action_horizon):
            idx = i + j
            if idx < len(dataset) and unwrapped_dataset[idx]['episode_index'] == unwrapped_dataset[i]['episode_index']:
                states.append(dataset[idx]['state'].numpy())
            else:
                states.append(state)
    else:
        logging.warning(f"Prompt '{prompt}' not found in dataset. Using zero state and images.")

    policy = create_policy(args)

    sample_input = {
        "observation/state": state,
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "prompt": prompt,
    }
    mask = np.zeros_like(actions).astype(bool)
    mask[:args.num_fixed_prefix] = True
    sample_edit = {
        "actions": actions,
        "mask": mask,
    }
    output = policy.sample(sample_input, sample_edit, num_samples=args.num_samples)
    for i in range(args.num_samples):
        motion = {
            'actions': output['actions'][i],
            'states': states,
            'text': prompt
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / f'{prompt}_{i}.npy', motion)
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True) 
    main(tyro.cli(Args))