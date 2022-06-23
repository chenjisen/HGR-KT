from run_all import *

model = 'HANKT1'  # HANKT1   LtDHKT   MyDHKT1
dataset = 'new_mini_09'  # new_mini_09   ednet   ass09   ass12

train(
    # cuda=False,
    model=model,
    dataset_name=dataset,
    random_seeds="0, 1, 20, 42, 666",
    # cross_validation=True,  **get_args(model_args, model)
    max_epochs=16, early_stop_patience=3,
)
