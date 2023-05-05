
# Fast Contextual Scene Graph Generation with Unbiased Context Augmentation

<hr style="height:1px;border:none;border-top:1px solid #555555;" /> 

| ![Real-Time Scene Graph Generation](https://live.staticflickr.com/65535/52872420698_539179cec2_m.jpg) |
<img src="https://live.staticflickr.com/65535/52872420698_539179cec2_m.jpg#pic_center" width="100">



This code contains three parts, corresponding to the three data sets of VG, PSG and OIDv6.

Basic operating requirement <pre><code>pip install -r requirements.txt</code></pre>


# Contents
## VG_dataset
Download the VG annotations [dataset](https://drive.google.com/file/d/19NEoW3oylGw0y2AA5iegocPHO6yxpb-f/view?usp=share_link)

Download the yolov5 object detection pre-trained model [yolol_object_test_28.pt](https://drive.google.com/file/d/1f77tyIHTyDyRBupzA7vz9S82oc7DZ8E3/view?usp=share_link)

Then put them into the VG_dataset folder
<pre><code> cd VG_dataset</code></pre>

<pre>
VG_dataset
    ├── ckn (contains trained ckn models)
    ├── vdn (contains trained vdn models)
    ├── evaluation 
    ├── models (yolov5 model)
    ├── dataset (annotations from VG datasets)
    ├── utils 
    ├── eval (contains eval files)
    ├── ckn_main.py
    ├── datapath.py
    ├── dataset.py
    ├── yolo_dataset.py
    └── yolol_object_test_28.pt
</pre>

### Evalution
#### C-SGG PredCls evalution on VG

<pre><code> python eval/predcls_ckn_val.py </code></pre>

#### C-SGG SGGen evalution on VG
<pre><code> python eval/sggen_ckn_val.py </code></pre>

#### CV-SGG PredCls evalution on VG

Download VG images from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

Rewrite the _**'image_file'**_ in datapath.py according to download images'path

<pre><code> python eval/predcls_vdn_val.py </code></pre>

#### CV-SGG SGGen evalution on VG
<pre><code> python eval/sggen_vdn_val.py </code></pre>

### Train C-SGG
During the verification process, the loaded annotations will be saved as npy files for subsequent fast loading. Before training, the npy file of the test set needs to be deleted.
<pre><code> rm *.npy *.pkl </code></pre>

Start training
<pre><code> python ckn_main.py </code></pre>

#### Acknowledgment: 
The VG dataset and evalutation are from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

The yolov5 model are trained based on [yolov5](https://github.com/ultralytics/yolov5)


<hr style="height:1px;border:none;border-top:1px solid #555555;" />

## PSG_dataset
Download the PSG annotations [dataset](https://drive.google.com/file/d/1FgnPMumTNeFM-1HLOQTVqCHJh8ozeVuj/view?usp=share_link)

Download the panoptic segmentation results [SegFromer_PVTV](https://drive.google.com/file/d/1E4AuyJRaG0N1Osl-Hi7NUpiZ1bZNa6mY/view?usp=share_link) for context description.

Download the groudtruth [psg_eval_results.pytorch](https://drive.google.com/file/d/1m7g4ubnUhrIQw4BfQ98aGSpq-oOvXjK5/view?usp=share_link) for evaluation.

Then put them into the PSG_dataset folder
<pre><code> cd PSG_dataset</code></pre>
<pre>
PSG_dataset
    ├── ckn (contains trained ckn models)
    ├── vdn  (contains trained vdn models)
    ├── dataset (annotations from OpenPSG datasets)
    ├── openpsg (from [OpenPSG] code repository)
    ├── psg_eval_results.pytorch (contains groudtruth for facilitate evaluation)
    ├── SegFromer_PVTV (contains panoptic segmentation results)
    ├── psg_infer.py 
    ├── psg_visual_infer.py
    ├── relation.json
    └── sgg_eval.py
</pre>

#### C-SGG SGGen evalution on OpenPSG

<pre><code>python psg_infer.py</code></pre>
Under the /PSG_dataset folder will generation a new file *relation.json*

<pre><code>python sgg_eval.py</code></pre>
Evaluate newly generated results. 
Due to the processed grondtruth from *[OpenPSG](https://github.com/Jingkang50/OpenPSG) /tools/grade.py*, the computer memory is preferably >32GB


#### CV-SGG SGGen evalution on OpenPSG
Download the images from [OpenPSG](https://github.com/Jingkang50/OpenPSG)

Rewrite the _**'image_path'**_ in psg_visual_infer.py according to dataset path
<pre><code>python psg_visual_infer.py #infer</code></pre>
<pre><code>python sgg_eval.py #evalute</code></pre>



#### Acknowledgment: 
The annotation, images, and groudtruth are from [OpenPSG](https://github.com/Jingkang50/OpenPSG) 

The panoptic segmentation results are from [Panoptic SegFormer PVTv2-B5](https://github.com/zhiqi-li/Panoptic-SegFormer)

<hr style="height:1px;border:none;border-top:1px solid #555555;" /> 

## OID_dataset

Download the [openimage_v6_test](https://drive.google.com/file/d/1Dkx7-ioEVffPeezeeQg8bhzo9j_8ROJM/view?usp=sharing) for object detection results and groudtruth. 

Then put it into the OID_dataset folder.
<pre><code> cd OID_dataset</code></pre>
<pre>
OID_dataset
    ├── ckn   (contains trained ckn models)
    ├── openimage_v6_test (for facilitate evaluation)
    ├── pysgg (from [PySGG] code repository)
    ├── vdn   (contains trained vdn models)
    ├── utils_evaluation.py
    ├── oid_inference.py
    ├── oid_visual_inference.py
    └── oid_evaluation.py
</pre>

#### C-SGG SGGen evalution on OIDv6

<pre><code>python oid_inference.py</code></pre>
Under the /openimage_v6_test folder will generation a new file *eval_results.pytorch*

<pre><code>python oid_evaluation.py</code></pre>
Evaluate newly generated results

#### CV-SGG SGGen evalution on OIDv6
Download the processed OpenImagev6 dataset from [PySGG](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md)

Rewrite the _**'yourpath'**_ in oid_visual_inference.py according to dataset path
<pre><code>python oid_visual_inference.py</code></pre>
Under the /openimage_v6_test folder will generation a new file *eval_results.pytorch*

<pre><code>python oid_evaluation.py</code></pre>
Evaluate newly generated results

#### Acknowledgment: 
The processed datasets and object detection results are from [PySGG](https://github.com/SHTUPLUS/PySGG)

