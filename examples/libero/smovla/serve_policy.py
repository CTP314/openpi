import dataclasses
import enum
import logging
import pathlib
import socket

import tyro

import asyncio
import http
import time
import traceback
import numpy as np
import torch

import websockets.asyncio.server as _server
import websockets.frames
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import functools

import msgpack
import numpy as np

def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj

Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

logger = logging.getLogger(__name__)

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: SmolVLAPolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        device: str = "cuda:0",
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._device = device
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                state = torch.from_numpy(obs["observation/state"])
                base_image = torch.from_numpy(obs["observation/image"])
                wrist_image = torch.from_numpy(obs["observation/wrist_image"])
                task_description = obs["prompt"]
                
                state = state.to(torch.float32)
                base_image = base_image.to(torch.float32) / 255
                base_image = base_image.permute(2, 0, 1)
                wrist_image = wrist_image.to(torch.float32) / 255
                wrist_image = wrist_image.permute(2, 0, 1)

                # Send data tensors from CPU to GPU
                state = state.to(self._device, non_blocking=True)
                base_image = base_image.to(self._device, non_blocking=True)
                wrist_image = wrist_image.to(self._device, non_blocking=True)
                
                state = state.unsqueeze(0)  # Add extra batch dimension
                base_image = base_image.unsqueeze(0)  # Add extra batch dimension
                wrist_image = wrist_image.unsqueeze(0)  # Add extra batch dimension
                
                observation = {
                    "observation.images.image": base_image,
                    "observation.images.wrist_image": wrist_image,
                    "observation.state": state,
                    "task": task_description
                }


                infer_time = time.monotonic()
                actions = []
                for _ in range(self._policy.config.n_action_steps):
                    action = self._policy.select_action(observation)
                    action_numpy = action.squeeze(0).to("cpu").numpy()
                    actions.append(invert_gripper_action(normalize_gripper_action(action_numpy, binarize=False)))
                action = dict(actions=np.array(actions))
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

@dataclasses.dataclass
class Args:
    pretrained_policy_path: pathlib.Path
    ckpt_path: pathlib.Path | None
    port: int = 8000
    device: str = "cuda:0"

def main(args: Args) -> None:
    policy = SmolVLAPolicy.from_pretrained(args.pretrained_policy_path).to(args.device)
    if args.ckpt_path is not None:
        policy.load_state_dict(torch.load(args.ckpt_path)["model_state_dict"])
        logging.info("Loaded policy weights from %s", args.ckpt_path)
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        device=args.device,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
