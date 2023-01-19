## Decomposing 3D Scenes into Object via Unsupervised Volume Segmentation

This codebase contains the official code for the [ObSuRF model](https://stelzner.github.io/obsurf/) by Karl Stelzner, Kristian Kersting, and Adam R. Kosiorek.

### Dependencies
The dependencies for the model may be installed using the following commands, starting from a recent
(python 3.7 or newer) Anaconda installation.
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tensorboardx opencv-python pyyaml imageio matplotlib tqdm
```

### Datasets
To download the datasets, check the [project page](https://stelzner.github.io/obsurf/). The unpacked data should
be placed (or symlinked) in the `./data/` directory. For notes on how to use custom data, see [this comment](https://github.com/stelzner/obsurf/blob/1089a275bf6040c2ba2e1c9c75c369657f7a5c7c/obsurf/data.py#L95).

### Run
The model is run by executing `python train.py runs/[dataset]/[model]/config.yaml`. Visualizations,
logs, and checkpoints are stored in the directory of the specified config file.

Novel view sequences may be obtained from a trained model via `render.py`, e.g.  
`python render.py runs/[dataset]/[model]/config.yaml --sceneid 0`. Sequences may be compiled into
videos using `compile_video.py`.

### Resource Requirements
The model was trained using 4 A100 GPUs with 40GB of VRAM, each. If you do not have these resources available, consider editing the config files to reduce the batch size (`training: batch_size`) or the number of target points per scene (`data: num_points`). We have not found the model to be particularly sensitive to either value.

### Checkpoints
Here, we provide some pretrained checkpoints for reference. 

| Model | Dataset | Link |
| ---|---|---|
|ObSuRF with pixel conditioning and overlap loss | CLEVR3D | [Link](https://drive.google.com/file/d/1A1PUYw4GoacH59N_pqaz-n1cq_Aa-BKe/view?usp=sharing)
|ObSuRF with pixel conditioning and overlap loss | MultiShapeNet | [Link](https://drive.google.com/file/d/1Yo9T49buQGjFhAzNQ1trf_EYajNLmLYr/view?usp=sharing)



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
 * https://github.com/autonomousvision/occupancy_networks
 * https://github.com/deepmind/multi_object_datasets
 * https://github.com/bmild/nerf
