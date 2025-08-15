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

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

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
        policy: DiffusionPolicy,
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
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                state = torch.from_numpy(obs["observation/state"])
                base_image = torch.from_numpy(obs["observation/image"])
                wrist_image = torch.from_numpy(obs["observation/wrist_image"]) 
                
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
                    "observation.images.base": base_image,
                    "observation.images.wrist": wrist_image,
                    "observation.state": state,
                }


                infer_time = time.monotonic()
                actions = []
                for _ in range(self._policy.config.n_action_steps):
                    action = self._policy.select_action(observation)
                    actions.append(action.squeeze(0).to("cpu").numpy())
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
    port: int = 8000
    device: str = "cuda:0"

def main(args: Args) -> None:
    policy = DiffusionPolicy.from_pretrained(args.pretrained_policy_path).to(args.device)
    
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
