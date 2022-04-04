## Decomposing 3D Scenes into Object via Unsupervised Volume Segmentation

This codebase contains the code for the ObSuRF model. We highlight the following files:

 * `train.py` contains the training loop
 * `obsurf/model/training.py` contains the code for running training, visualization, and evaluation steps
 * `obsurf/model/representations.py` specifies how volumes are parameterized and composed
 * `obsurf/encoder.py` contains the encoder model
 * `obsurf/model/decoder.py` and `obsurf/model/decoder_modules.py` contain the decoder model
 * `obsurf/data.py` contains the data loading routines
 * `runs/[dataset]/[model]/config.yaml` contain the configurations used for the experiments in the paper

### Dependencies
The dependencies for the model may be installed using the following commands, starting from a recent
(python 3.7 or newer) Anaconda installation.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tensorboardx opencv-python
```

### Datasets
To download the datasets, check [https://stelzner.github.io/obsurf/]. The unpacked data should
be placed (or symlinked) in the `./data/` directory.

### Run
The model is run by executing `python train.py runs/[dataset]/[model]/config.yaml`. Visualizations,
logs, and checkpoints are stored in the directory of the specified config file.

Novel view sequences may be obtained from a trained model via `render.py`, e.g.  
`python render.py runs/[dataset]/[model]/config.yaml --sceneid 0`. Sequences may be compiled into
videos using `compile_video.py`.


### Citation
If you found this codebase useful, please consider citing our paper:

```
@article{stelzner2021decomposing,
  title={Decomposing 3d scenes into objects via unsupervised volume segmentation},
  author={Stelzner, Karl and Kersting, Kristian and Kosiorek, Adam R},
  journal={arXiv:2104.01148},
  year={2021}
}
```

### Acknowledgement
Parts of this codebase were adapted from the following publicly available repositories:
 * [https://github.com/autonomousvision/occupancy_networks]
 * [https://github.com/deepmind/multi_object_datasets]
 * [https://github.com/bmild/nerf]
