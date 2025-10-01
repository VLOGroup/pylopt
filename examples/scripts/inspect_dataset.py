import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from pylopt.dataset.ImageDataset import TrainingImageDataset
from pylopt.utils.seeding_utils import seed_random_number_generators
from pylopt.dataset.dataset_utils import collate_function

def main():
    root_path = '/home/florianthaler/Documents/data/image_data/BSDS300/images/train'
    crop_size = 64
    batch_size = 8
    dataset = TrainingImageDataset(root_path=root_path)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=lambda x: collate_function(x, crop_size=crop_size))

    batch_0 = next(iter(loader))
    fig = plt.figure()
    for i, item in enumerate(batch_0):
        ax = fig.add_subplot(1, batch_size, i + 1)
        ax.imshow(item.squeeze().detach().cpu().numpy())
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.show()

if __name__ == '__main__':
    seed_random_number_generators(seed_val=123)
    main()