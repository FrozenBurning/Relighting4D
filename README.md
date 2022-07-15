<div align="center">

<h1>Relighting4D: Neural Relightable Human from Videos</h1>

<div>
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<strong><a href='https://eccv2022.ecva.net/' target='_blank'>ECCV 2022</a></strong>

### [Project Page](https://frozenburning.github.io/projects/relighting4d) | [Video](https://youtu.be/NayAw89qtsY) | [Paper](https://arxiv.org/abs/2207.07104)

<tr>
    <img src="https://github.com/FrozenBurning/FrozenBurning.github.io/blob/master/projects/relighting4d/img/teaser.gif" width="100%"/>
</tr>
</div>

## Citation

If you find our work useful for your research, please consider citing this paper:

```
@inproceedings{chen2022relighting,
    title={Relighting4D: Neural Relightable Human from Videos},
    author={Zhaoxi Chen and Ziwei Liu},
    booktitle={ECCV},
    year={2022}
}
```


## Installation
We recommend using [Anaconda](https://www.anaconda.com/) to manage your python environment. You can setup the required environment by the following command:
```(bash)
conda env create -f environment.yml
conda activate relighting4d
```

## Datasets

### People-Snapshot
We follow [NeuralBody](https://github.com/zju3dv/neuralbody) for data preparation.
1. Download the People-Snapshot dataset [here](https://graphics.tu-bs.de/people-snapshot).
2. Process the People-Snapshot dataset using the [script](tools/process_snapshot.py).
3. Create a soft link:

    ```(bash)
    cd /path/to/Relighting4D
    mkdir -p data
    cd data
    ln -s /path/to/people_snapshot people_snapshot
    ```

### ZJU-MoCap
Please refer to [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md) for requesting the download link. Once downloaded, don't forget to add a soft link:
```(bash)
cd /path/to/Relighting4D
mkdir -p data
cd data
ln -s /path/to/zju_mocap zju_mocap
```

## Training
We first reconstruct an auxiliary density field in Stage I and then train the whole pipeline in Stage II. All trainings are done on a Tesla V100 GPU with 16GB memory.

Take the training on `female-3-casual` as an example.

* Stage I:
    ```(bash)
    python train_net.py --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c resume False gpus "0,"
    ```
    The model weights will be saved to `/data/trained_model/if_nerf/female3c/latest.pth`.

* Stage II:
    ```(bash)
    python train_net.py --cfg_file configs/snapshot_exp/snapshot_f3c.yaml task relighte2e exp_name female3c_relight train_relight True resume False train_relight_cfg.smpl_model_ckpt ./data/trained_model/if_nerf/female3c/latest.pth gpus "0,"
    ```
    The final model will be saved to `/data/trained_model/relighte2e/female3c_relight/latest.pth`.

* Tensorboard:
    ```
    tensorboard --logdir data/record/if_nerf
    tensorboard --logdir data/record/relighte2e
    ```


## Rendering
To relight a human performer from the trained video, our model requires an HDR environment map as input. We provide 8 HDR maps at [light-probes](light-probes/). You can also use your own HDRIs or download some samples from [Poly Haven](https://polyhaven.com/hdris).

Here, we take the rendering on `female-3-casual` as an example. 

* Relight with novel views of single frame
    ```(bash)
    python run.py --type relight --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c_relight task relighte2e vis_relight True ratio 0.5 gpus "0,"
    ```

* Relight the dynamic humans in video frames
    ```(bash)
    python run.py --type relight_npose --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c_relight task relighte2e vis_relight True vis_relight_npose True ratio 0.5 pyramid False gpus "0,"
    ```
The results of rendering are located at `/data/render/`. For example, rendering results with [courtyard HDR environment](light-probes/courtyard.hdr) are shown as follows:

<table>
<tr>
    <td align='center' width='50%'><img src="https://frozenburning.github.io/projects/relighting4d/img/nview.gif" width="100%"/></td>
    <td align='center' width='50%'><img src="https://frozenburning.github.io/projects/relighting4d/img/npose.gif" width="100%"/></td>
</tr>
</table>

## Acknowledgements
This work is supported by the National Research Foundation, Singapore under its AI Singapore Programme, NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), and under the RIE2020 Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

Relighting4D is implemented on top of the [NeuralBody](https://github.com/zju3dv/neuralbody) codebase.
