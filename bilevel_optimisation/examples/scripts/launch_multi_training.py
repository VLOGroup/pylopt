import os
import torch
import subprocess
import multiprocessing
import logging
import time

from bilevel_optimisation.utils.LoggingUtils import setup_logger

DENOISING_TRAIN_SCRIPT = 'denoising_train.py'

def run_training_script_on_gpu(config_queue: multiprocessing.Queue,  gpu_id: int):
    while not config_queue.empty():
        try:
            config_dir = config_queue.get_nowait()
        except:
            break

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        time.sleep(3)
        logging.info('[MULTI-TRAIN][GPU_{:d}] Run {:s}'.format(gpu_id, DENOISING_TRAIN_SCRIPT))

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DENOISING_TRAIN_SCRIPT)
        cmd = ['python', script_path, '--configs', config_dir]
        subprocess.run(cmd, env=env)

        logging.info('[MULTI-TRAIN][GPU_{:d}] Finished {:s}'.format(gpu_id, DENOISING_TRAIN_SCRIPT))

def main():
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

    custom_config_dir_list = ['multi_training_I', 'multi_training_II', 'multi_training_III', 'multi_training_IV']
    config_dir_queue = multiprocessing.Queue()
    for cfg_dir in custom_config_dir_list:
        config_dir_queue.put(cfg_dir)

    procs = []
    for gpu_id in range(0, len(available_gpus)):
        p = multiprocessing.Process(target=run_training_script_on_gpu, args=(config_dir_queue, gpu_id))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == '__main__':
    log_dir_path = './data'
    setup_logger(data_dir_path=log_dir_path, log_level_str='info')
    main()
