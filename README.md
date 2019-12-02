This is the official implementation of the paper "Adversarial Autoencoders with Constant-Curvature Latent Manifolds" by D. Grattarola, C. Alippi, and L. Livi. (2018, [https://arxiv.org/abs/1812.04314](https://arxiv.org/abs/1812.04314)).  

This code showcases the general structure of the methodology used for the experiments in the paper, and allows to reproduce the results on MNIST (the other two applications are conceptually similar, but the code was much more messy). 

Please cite the paper if you use any of this code for your research:   

```
@article{grattarola2019adversarial,
  title={Adversarial autoencoders with constant-curvature latent manifolds},
  author={Grattarola, Daniele and Livi, Lorenzo and Alippi, Cesare},
  journal={Applied Soft Computing},
  volume={81},
  pages={105511},
  year={2019},
  publisher={Elsevier}
}
```

## Setting up

The code is implemented for Python 3 and tested on Ubuntu 16.04.  
To run the code, you will need to have the following libraries installed on your system:

- [Keras](https://keras.io/) (`pip install keras`), a high-level API for deep learning;
- [Spektral](https://danielegrattarola.github.io/spektral/) (`pip install spektral`), a Keras extension to build graph neural networks;
- [CDG](https://github.com/dan-zam/cdg) (see README on Github), a library for non-Euclidean geometry. 

The code also depends on Numpy and Scikit-learn.  

## Running experiments

The `src` folder includes a script to run the algorithm proposed in the paper on MNIST. 
After installing the necessary libraries, simply run: 
```bash
$ python src/mnist.py
```
to train the CCM-AAE and run the semi-supervised classification.

