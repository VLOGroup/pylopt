import os
from concurrent.futures import ProcessPoolExecutor

def run_training_script_on_gpu(config_dir_name: str, gpu_id: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str()
    os.system()


if __name__ == '__main__':
    config_dir_name_list = ['example_training_I', 'example_training_II']
    gpu_id_list = []

    with ProcessPoolExecutor(max_workers=len(gpu_id_list)) as executor:
        futures = []

        for i, config_dir in enumerate(config_dir_name_list):
            gpu_id = gpu_id_list[i % len(gpu_id_list)]
            futures.append()
