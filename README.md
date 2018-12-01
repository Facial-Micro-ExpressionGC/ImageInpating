# ImageInpating

This project implements Image Inpaiting using Deep learning with Generative Adversial Networks.

It uses [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset to reconstruct the center part of the image.

## Required Packages

All `conda` packages     described on `requirements.txt`

## Usage

- Run `main.py` to train the model. If you have saved models, add `--load-model` argument to load them

## Maintenance

Maintainers: Marcelo Andrade <marceloga1@al.insper.edu.br>

Tickets: Can be opened in Github Issues.

## References

[This](https://github.com/eriklindernoren/Keras-GAN) repository was used as a boilerplate to create these models.

Some hyperparameters were changed according to [this](https://arxiv.org/abs/1604.07379) paper.

## Next Steps

- [ ] Implement code on [Places](http://places.csail.mit.edu/) dataset.

## License

This project is licensed under MIT license - see [LICENSE.md](LICENSE.md) for more details.
