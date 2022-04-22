import logging

import fire

from benchmark.train import train, grid_search

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.INFO)


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'grid_search': grid_search,
    })
