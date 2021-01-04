# Colored MNIST

### Instructions
1. Generate Colored MNIST train data, test data, and confound test data: ```python generate_colored_mnist.py```
2. Train a standard Conditional VAE and create a new Intervened VAE train data: ```python train_vae.py```
3. Evaluate a simple classifier using different dataset: ```python evaluate_cnn_classification.py```

For IRM:

1. Set paths in ```generate-irm-data.py``` and ```main-irm.py```
2. Generate colored MNIST data using ```generate-irm-data.py```
2. Run IRM optimization using ```irm_reproduce.sh```

Please change the directory path inside the code to your own environment. 
