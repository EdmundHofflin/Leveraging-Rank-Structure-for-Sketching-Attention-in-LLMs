# Leverage Rank Structure for Sketching Attention in LLMs

We provide a framework for employing matrix sketching in the attention-value matrix multiplication within the attention mechanisms of large language models. The framework allows for the sketching method to be adapted with ease, with many options already available, and then tested with various models. Plotting functionality of the performance and various internal metrics of the matrix sketch are also available.

## Setup

The requirements for an [Anaconda](https://www.anaconda.com/download) environment are found in ```sketched_attention.yml```. They can be used as follows:

```bash
conda env create --file sketched_attention.yml
```

and the environment activated like so:

```bash
conda activate sketched_attention
```

Depending upon your operating system, you may need to force an update for the Brotli package:

```bash
conda install brotli --force_reinstall
```

## Testing

The available models are listed in ```src/models/model_config```. Although not necessary, we recommend using ```src/model_import.py```, which will download and save the models and also test basic functionality of this framework. Otherwise, the framework is run using ```src/testing.py```. Selection of the framework's settings, from sketching functionality to file management to model and data choice, is done by passing arguments, all of which can be viewed by using ```--help```. Alternatively, a config ```.json``` file can also be used to save configurations and pass them easily. ```/src/config/``` has various examples.

## Plotting

Plotting is completed through ```src/plot.py```. It too has arguments that can be changed, but to generate standard plots, the only argument needed is ```--config $NAME```, where ```$NAME``` is the name given to the test configuration already run. Other arguments can be viewed by using ```--help```. 