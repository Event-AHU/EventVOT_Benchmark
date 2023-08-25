<div align="center">

<img src="https://github.com/Event-AHU/EventVOT_Benchmark/blob/main/figures/EventVOT_white.png" width="600">
  
**The First High Definition (HD) Event based Visual Object Tracking Benchmark Dataset** 

------

<p align="center">
  • <a href="">arXiv</a> • 
  <a href="">Baselines</a> •
  <a href="">DemoVideo</a> • 
  <a href="">Tutorial</a> •
</p>

</div>



# :dart: Abstract 
Tracking using bio-inspired event cameras draws more and more attention in recent years. Existing works either utilize aligned RGB and event data for accurate tracking or directly learn an event-based tracker. The first category needs more cost for inference and the second one may be easily influenced by noisy events or sparse spatial resolution. In this paper, we propose a novel hierarchical knowledge distillation framework that can fully utilize multi-modal / multi-view information during training to facilitate knowledge transfer, enabling us to achieve high-speed and low-latency visual tracking  during testing by using only event signals. Specifically, a teacher Transformer  based multi-modal  tracking framework is first trained  by feeding the RGB frame and event stream simultaneously. Then, we design a new hierarchical knowledge distillation strategy which includes pairwise similarity, feature representation and response maps based knowledge distillation to guide the learning of the student Transformer network. Moreover, since existing event-based tracking datasets are all low-resolution ($346 \times 260$), we propose the first large-scale high-resolution ($1280 \times 720$) dataset named EventVOT. It contains 1141 videos and covers a wide range of categories such as pedestrians, vehicles, UAVs, ping pongs, etc. Extensive experiments on both low-resolution (FE240hz, VisEvent, COESOT), and our newly proposed high-resolution EventVOT dataset fully validated the effectiveness of our proposed method. 


# :collision: Update Log 



# :video_camera: Demo Video
A demo video can be found by clicking the image below: 
<p align="center">
  <a href="https://youtu.be/FcwH7tkSXK0">
    <img src="https://github.com/Event-AHU/EventVOT_Benchmark/blob/main/figures/EventVOT_youtube.png" alt="DemoVideo" width="800"/>
  </a>
</p> 


# :hammer: Environment 

**Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline**
<p align="center">
  <img width="85%" src="./figures/HDETrack.jpg" alt="Framework"/>
</p>


Install env
```
conda create -n hdetrack python=3.8
conda activate hdetrack
bash install.sh
```

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

Then, put the tracking datasets EventVOT in `./data`. 

Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$/pretrained_models`

Download the model weights and put it on `$/output/checkpoints/train/hdetrack`


## Train & Test & Evaluation
```
# train
python tracking/train.py --script hdetrack --config hdetrack_eventvot --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0

# test
python tracking/test.py hdetrack hdetrack_eventvot --dataset eventvot --threads 1 --num_gpus 1

# eval
python tracking/analysis_results.py --dataset eventvot --parameter_name hdetrack_eventvot
```




### Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX 3090 GPU.

```
# Profiling ceutrack_coesot
python tracking/profile_model.py --script hdetrack --config hdetrack_eventvot
```



# :dvd: EventVOT Dataset 


* **Event Image version** (train.zip 28.16GB, val.zip 703M, test.zip 9.94GB)

:floppy_disk: **Baidu Netdisk**: link：https://pan.baidu.com/s/1NLSnczJ8gnHqF-69bE7Ldg?pwd=wsad code：wsad


* **Complete version** (Event Image + Raw Event data, train.zip 180.7GB, val.zip 4.34GB, test.zip 64.88GB)
  
:floppy_disk: **Baidu Netdisk**: link：https://pan.baidu.com/s/1ZTX7O5gWlAdpKmd4R9VhYA?pwd=wsad code：wsad
  
:floppy_disk: **Dropbox**: https://www.dropbox.com/scl/fo/fv2e3i0ytrjt14ylz81dx/h?rlkey=6c2wk2z7phmbiwqpfhhe29i5p&dl=0

The directory should have the below format:
```Shell
├── EventVOT dataset
    ├── Training Subset (841 videos, 180.7GB)
        ├── recording_2022-10-10_17-28-38
            ├── img
            ├── recording_2022-10-10_17-28-38.csv
            ├── groundtruth.txt
            ├── absent.txt
        ├── ... 
    ├── Testing Subset (282 videos, 64.88GB)
        ├── recording_2022-10-10_17-28-24
            ├── img
            ├── recording_2022-10-10_17-28-24.csv
            ├── groundtruth.txt
            ├── absent.txt
        ├── ...
    ├── validating Subset (18 videos, 4.34GB)
        ├── recording_2022-10-10_17-31-07
            ├── img
            ├── recording_2022-10-10_17-31-07.csv
            ├── groundtruth.txt
            ├── absent.txt
        ├── ... 
```


# :triangular_ruler: Evaluation Toolkit

1. unzip the EventVOT_eval_toolkit.zip, and open it with Matlab (over Matlab R2020).
2. add your tracking results and [baseline results (Passcode：wsad)](https://pan.baidu.com/s/1cvWg7cw1_kLjBMP7y7bEbg)  in `$/eventvot_tracking_results/` and modify the name in `$/utils/config_tracker.m`
3. run `Evaluate_EventVOT_benchmark_SP_PR_only.m` for the overall performance evaluation, including SR, PR, NPR.
4. run `plot_BOC.m` for BOC score evaluation and figure plot.
5. run `plot_radar.m` for attributes radar figrue plot.
<p align="center">
  <img width="45%" src="./figures/attributes.png" alt="Radar"/><img width="55%" src="./figures/BOC.png" alt="Radar"/>
</p>
7. run `Evaluate_EventVOT_benchmark_attributes.m` for attributes analysis and figure saved in `$/res_fig/`.


# :chart_with_upwards_trend: Benchmark Results
<p align="left">
  <img width="100%" src="./figures/SRPRNPR.png" alt="SRPRNPR"/>
</p>


# :cupid: Acknowledgement 
* Thanks for the  [CEUTrack](https://github.com/Event-AHU/COESOT), [OSTrack](https://github.com/botaoye/OSTrack), [PyTracking](https://github.com/visionml/pytracking) and [ViT](https://github.com/rwightman/pytorch-image-models) library for a quickly implement.

# :newspaper: Citation 
```bibtex
@article{wang2023eventvot,
  title={Event Stream-based Visual Object Tracking: A High-Resolution Benchmark Dataset and A Novel Baseline},
  author={},
  journal={},
  year={2023}
}
```


































