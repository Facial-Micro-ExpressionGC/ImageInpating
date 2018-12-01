import argparse

import utils as utils
from models.model import ModelGAN

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--load-model",
                    action='store_true',
                    help="Loads previous saved models",
                    default=False)
args = parser.parse_args()
load_model = args.load_model

data, _ = utils.load_cifar10()

model = ModelGAN(data)
if load_model:
    model.load_model()

model.compile_models()
model.train()