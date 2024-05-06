# from https://github.com/OpenGVLab/InternImage/blob/master/segmentation/test.py
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmcv.runner import get_dist_info
import mmcv
import torch
from mmseg.ops import resize
from functools import partial

class ExtractorModel(torch.nn.Module):
    def __init__(self, extractor, model) -> None:
        super().__init__()
        self.extractor = extractor
        self.model = model
        self.num_classes = 150
        self.test_cfg = {'mode': 'slide', 'crop_size': (512, 512), 'stride': (341, 341)}
    
    def forward(self, img, img_metas, return_loss=False):
        if self.extractor is None:
            return self.model(img=img, img_metas=img_metas, return_loss=return_loss)
        else:
            img[0] = img[0]
            metas = [a.data[0] for a in img_metas]
            # print(metas)
            seg_logit = self.slide_inference(img[0], img_meta=metas[0], rescale=True)
            output = torch.nn.functional.softmax(seg_logit, dim=1)
            flip = metas[0][0]['flip']
            if flip:
                flip_direction = metas[0][0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            seg_pred = list(seg_pred)
            return seg_pred
    
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                # print(crop_img.shape)
                crop_seg_logit = self.extractor.extract_feature(self.model, crop_img, img_meta)
                preds += torch.nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False,
                warning=False)
        return preds

test_data_config = {
    'type': 'ADE20KDataset', 
    'data_root': 'data/ADEChallengeData2016', 
    'img_dir': 'images/validation', 
    'ann_dir': 'annotations/validation', 
    'pipeline': [
        {'type': 'LoadImageFromFile'}, 
        {'type': 'MultiScaleFlipAug', 
         'img_scale': (2048, 512), 
         'flip': False, 
         'transforms': [
            {'type': 'Resize', 'keep_ratio': True}, 
            {'type': 'ResizeToMultiple', 'size_divisor': 32}, 
            {'type': 'RandomFlip'}, 
            {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, 
            {'type': 'ImageToTensor', 'keys': ['img']}, 
            {'type': 'Collect', 'keys': ['img']}
        ]}
    ], 
    'test_mode': True}


def eval_ade20k(args, model, extractor):
    dataset = build_dataset(test_data_config)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False)

    m = ExtractorModel(extractor, model)

    results = single_gpu_test(
        m,
        data_loader,
        show=False,
        out_dir=args.show_dir,
        efficient_test=False,
        opacity=0.5,
        pre_eval=True,
        format_only=False,
        format_args={})

    rank, _ = get_dist_info()
    if rank == 0:
        eval_kwargs = dict(metric=['mIoU'])
        metric = dataset.evaluate(results, **eval_kwargs)
        metric_dict = dict(arch=args.arch, 
                           head_type=args.head_type, 
                           head_scale_count=args.head_scale_count,
                           dx=args.dx, 
                           dy=args.dy,
                           feature_extractor=args.feature_extractor, 
                           metric=metric)
        mmcv.dump(metric_dict, args.metric_output, indent=4)