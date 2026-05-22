import logging
import os
import sys
import os.path as osp
import datetime

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"), mode='a')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, f"test_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger