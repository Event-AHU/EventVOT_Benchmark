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



# Abstract 
Tracking using bio-inspired event cameras draws more and more attention in recent years. Existing works either utilize aligned RGB and event data for accurate tracking or directly learn an event-based tracker. The first category needs more cost for inference and the second one may be easily influenced by noisy events or sparse spatial resolution. In this paper, we propose a novel hierarchical knowledge distillation framework that can fully utilize multi-modal / multi-view information during training to facilitate knowledge transfer, enabling us to achieve high-speed and low-latency visual tracking  during testing by using only event signals. Specifically, a teacher Transformer  based multi-modal  tracking framework is first trained  by feeding the RGB frame and event stream simultaneously. Then, we design a new hierarchical knowledge distillation strategy which includes pairwise similarity, feature representation and response maps based knowledge distillation to guide the learning of the student Transformer network. Moreover, since existing event-based tracking datasets are all low-resolution ($346 \times 260$), we propose the first large-scale high-resolution ($1280 \times 720$) dataset named EventVOT. It contains 1141 videos and covers a wide range of categories such as pedestrians, vehicles, UAVs, ping pongs, etc. Extensive experiments on both low-resolution (FE240hz, VisEvent, COESOT), and our newly proposed high-resolution EventVOT dataset fully validated the effectiveness of our proposed method. 



# Demo Video
A demo video can be found by clicking the image below: 
<p align="center">
  <a href="https://youtu.be/FcwH7tkSXK0">
    <img src="https://github.com/Event-AHU/EventVOT_Benchmark/blob/main/figures/EventVOT_youtube.png" alt="DemoVideo" width="800"/>
  </a>
</p>



# EventVOT Dataset 


* **Event Image version** (train.zip 28.16GB, val.zip 703M, test.zip 9.94GB)
** **Baidu Netdisk**: link：https://pan.baidu.com/s/1NLSnczJ8gnHqF-69bE7Ldg?pwd=wsad code：wsad


* **Complete version** (Event Image + Raw Event data, train.zip 180.7GB, val.zip 4.34GB, test.zip 64.88GB)
** **Baidu Netdisk**: link：https://pan.baidu.com/s/1ZTX7O5gWlAdpKmd4R9VhYA?pwd=wsad code：wsad
** **Dropbox**: https://www.dropbox.com/scl/fo/fv2e3i0ytrjt14ylz81dx/h?rlkey=6c2wk2z7phmbiwqpfhhe29i5p&dl=0




# 
