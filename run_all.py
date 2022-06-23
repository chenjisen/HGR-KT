print('importing pytorch... ', end='')

import torch

print(torch.__version__)

print('importing pytorch lightning... ', end='')

import pytorch_lightning

print(pytorch_lightning.__version__)

from train import *

dataset_list = ['ednet', 'ass09', 'junyi', 'ass12']  #
models = ['HANKT1']  # ['MyDKT', 'HANKT1', 'LtDHKT', 'MyDHKT2']

if __name__ == '__main__':
    results = {}
    for dataset in ['new_mini_09'] + dataset_list:
        for model in models:
            result = train(
                model=model,
                dataset_name=dataset,
                cross_validation=True,
                random_seeds="0, 1, 20, 42, 666",
                max_epochs=1000, early_stop_patience=50,
                # **get_args(model_args, model)
            )

            results[model, dataset] = result

        print_results(models, sorted(dataset_list), results)
