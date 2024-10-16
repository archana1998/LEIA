# LEIA
Code for "LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation" (ECCV 2024) (Coming Soon)


## Create Environment

To setup an environment that runs this repository, follow the following instructions:

``` 
    conda create -n leia python=3.9
    conda activate leia
    pip install -r requirements.txt
```
Then install the torch bindings for [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):

```pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch```

This code has been tested with PyTorch 2.0.1 and CUDA 11.7.

## Data Processing

We used the [SAPIEN](https://sapien-sim.github.io/docs/user_guide/getting_started/installation.html) simulator interfaced with Python to render images from [PartNet-Mobility](https://sapien.ucsd.edu/browse) Dataset.

A helper Python script is provided in the repository to process data, but we also release pre-processed data here (TODO).

(TODO: add python script)
``` python utils/sapien_data.py ```


## Training

To train the model, use the following command:

``` python launch.py --config configs/nerf-blender-leia.yaml --train```

Hyperparameters can be modified by command line arguments, for details refer to the .yaml file in ```configs/```

To test the model,

```python launch.py --test --config=$TRAINED_MODEL_CONFIG --resume=$TRAINED_MODEL_CHECKPOINT trainer.limit_test_batches=5```

## Citation
```
@misc{swaminathan2024leialatentviewinvariantembeddings,
        title={LEIA: Latent View-invariant Embeddings for Implicit 3D Articulation}, 
        author={Archana Swaminathan and Anubhav Gupta and Kamal Gupta and Shishira R. Maiya and Vatsal Agarwal and Abhinav Shrivastava},
        year={2024},
        eprint={2409.06703},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2409.06703}, 
  }
```
## Acknowledgements

This code was inspired by [PARIS](https://github.com/3dlg-hcvc/paris), thanks to the authors!





