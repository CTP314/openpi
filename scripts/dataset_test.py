import logging
import platform
import jax
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

def init_logging():
    """自定义日志格式，使其更易读。"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.handlers[0].setFormatter(formatter)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"--- 正在以【数据集调试模式】运行 ---")
    logging.info(f"将在加载并检查第一批数据后退出。")
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"批处理大小 (Batch size) {config.batch_size} 必须能被设备数 {jax.device_count()} 整除。"
        )

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    logging.info("正在根据传入的 config 创建数据加载器...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,  
    )
    data_iter = iter(data_loader)

    logging.info("尝试从数据加载器中获取第一批数据...")
    try:
        batch = next(data_iter)
        logging.info("成功获取一批数据!")
    except StopIteration:
        logging.error("数据加载器为空，无法获取任何数据。请检查你的数据集路径和配置 (`--config.data_path`)。")
        return
    except Exception as e:
        logging.error(f"获取数据时发生错误: {e}")
        # 抛出异常以获得更详细的堆栈跟踪
        raise

    logging.info("=" * 80)
    logging.info("数据批次 (Batch) 结构、形状和类型:")
    logging.info(f"\n{training_utils.array_tree_to_info(batch)}")
    logging.info("=" * 80)

    if len(batch) == 2:
        observation, actions = batch
        logging.info("批次结构正确，包含 'observation' 和 'actions'。")
        if hasattr(observation, 'images') and isinstance(observation.images, dict):
            for cam_name, img_tensor in observation.images.items():
                logging.info(f"  - 摄像头 '{cam_name}' 图像形状: {img_tensor.shape}, 类型: {img_tensor.dtype}")
        logging.info(f"  - 动作 (actions) 形状: {actions.shape}, 类型: {actions.dtype}")

    logging.info("数据集调试完成，程序即将退出。")


if __name__ == "__main__":
    main(_config.cli())