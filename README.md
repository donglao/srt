# SRT: Stochastic Resonance transformers

## Visualization

visualization is provided in `visualization.ipynb`. You can change the `pretrain=mae` to get visualization for different model.

## Evaluatio: NYU-V2 depth. 
For evaluation of NYU-V2 depth we require the testset of NYU-V2 

**Step 1: Installing Dinov2 env**

Follow [Dinov2 repo](https://github.com/facebookresearch/dinov2) to install the dinov2-extra environment.

**Step 2: Prepare NYU-V2 testset**
```
cd data
ln -s /<path>/<to>/nyu_v2 .
```
Directory structure:
```
nyu_v2/
  testing/
    images/
      1.png
      2.png
      ...
    depths/
      1.png
      2.png
      ...
```

**Step 3: Run**

For dinov2 small with linear head, feature space ensemble
```
python depth_dinov2_main.py \
    --feature_extractor ensemble_dinov2_depth_feats \
    --do_eval \
    --data_path data/nyu_v2 \
    --dx 3 \
    --dy 3 \
    --head_type linear \
    --arch small
```
Change `--head_type` to be `{linear, dpt}` to use linear head or dpt head from dinov2. Change `--arch` to be `{small, base, large, giant}` to use different size ViT backbone. 


## Evaluation: DAVIS 2017 Video object segmentation
Please verify that you're using pytorch version 1.7.1 since we are not able to reproduce the results with most recent pytorch 1.8.1 at the moment.

**Step 1: Prepare DAVIS 2017 data**  
```
cd ..
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
cd SRT_icml/data
ln -s ../../davis-2017 .
```

**Step 2: Run Davis 2017 evaluation**  
```
python main.py \
    --data_path data/davis-2017-alt \
    --output_dir [dir] \
    --patch_size 16 \
    --arch vit_small \
    --feature_extractor ensemble_fast \
    --do_eval \
    --bs 1 \
    --dx 1 \
    --dy 1
```

**Step 3: Compute metrics**
```
git clone https://github.com/davisvideochallenge/davis2017-evaluation $HOME/davis2017-evaluation
python $HOME/davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path /path/to/saving_dir --davis_path $HOME/davis-2017/DAVIS/
```

## Evaluation: ADE20K semantic Segmenation

**Step 1: Install dinov2 env**

Follow [Dinov2 repo](https://github.com/facebookresearch/dinov2) to install the dinov2-extra environment.

**Step 2: Prepare ADE20K data**

Download ADEChallengeData2016 from [this link](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

```
cd data
ln -s /<path>/<to>/ADEChallenge2016 .
```

**Step 3: Run** 

For dinov2 small with linear head, feature space ensemble
```
python segment_dinov2_main.py \
    --feature_extractor ensemble_dinov2_seg_feats \
    --arch small \
    --head_type linear \
    --dx 3 \
    --dy 3 \
    --bs 36 \
    --do_eval \
    --metric_output dinov2-small-linear-feat-dxdy3.json 
```
Same as dpeth estimation, `--arch` can be changed to use different ViT. 

## Upcoming

A more comprehensive (containing other tasks) and organized repo will be released later. 