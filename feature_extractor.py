import torch
import torch.nn as nn
from torchvision import transforms
import math
import torch.nn.functional as F
from mmseg.ops import resize



class FeatureExtractor():
    def __init__(self, patch_size):
        self.patch_size = patch_size


    def extract_feature(self, model, frame, return_h_w=False):
        raise NotImplementedError()



class BasicExtractor(FeatureExtractor):
    def extract_feature(self, model, frame, return_h_w=False):
        """Extract one frame feature everytime."""
        out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
        out = out[:, 1:, :]  # we discard the [CLS] token
        h, w = int(frame.shape[1] / self.patch_size), int(frame.shape[2] / self.patch_size)
        dim = out.shape[-1]
        out = out[0].reshape(h, w, dim)
        out = out.reshape(-1, dim)
        if return_h_w:
            return out, h, w
        return out


class EnsembledExtractor(FeatureExtractor):
    def __init__(self, patch_size, dx, dy, bs):
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.dx = dx
        self.dy = dy
        self.bs = bs
        super().__init__(patch_size)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        frame = frame.unsqueeze(0).cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dx, self.dx + 1):
            for j in range(- self.dy, self.dy + 1):
                aug_list.append(transforms.functional.affine(frame, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b, _, h, w = image_batch.shape
        h, w = int(h / self.patch_size), int(w / self.patch_size)

        feature_field_full = None
        ind = 1
        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            out = model.get_intermediate_layers(image_batch[mb_start_idx:mb_end_idx], n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token

            mb = out.shape[0]

            dim = out.shape[-1]
            feature_field = out.reshape(mb, h, w, dim).permute(0,3,1,2)
            if feature_field_full is None:
                feature_field_full = torch.nn.functional.interpolate(feature_field[:1], scale_factor=self.patch_size, mode='nearest')

            for ind in range(mb_start_idx, mb_end_idx):
                if ind == 0:
                    continue
                i, j = ind_i_j[ind]
                new_feature_field = torch.nn.functional.interpolate(feature_field[ind-mb_start_idx:ind-mb_start_idx+1, :, :, :], scale_factor=self.patch_size, mode='nearest')
                feature_field_full += transforms.functional.affine(new_feature_field, translate=[-i,-j], angle=0, scale=1, shear=0)

        feature_fine_mean = feature_field_full / b
        out = self.avg_pool(feature_fine_mean).squeeze().permute(1,2,0)

        dim = out.shape[-1]
        out = out.reshape(-1, dim)

        if return_h_w:
            return out, h, w
        return out


class EnsembledExtractorFast(FeatureExtractor):
    def __init__(self, patch_size, dx, dy, bs):
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.dx = dx
        self.dy = dy
        self.bs = bs
        super().__init__(patch_size)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        frame = frame.unsqueeze(0).cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dy, self.dy + 1):
            for j in range(- self.dx, self.dx + 1):
                aug_list.append(transforms.functional.affine(frame, translate=[j,i], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b, _, h, w = image_batch.shape
        h, w = int(h / self.patch_size), int(w / self.patch_size)

        patch_pixs = self.patch_size ** 2
        ind = 1
        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            out = model.get_intermediate_layers(image_batch[mb_start_idx:mb_end_idx], n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token

            mb = out.shape[0]
            dim = out.shape[-1]
            feature_field = out.reshape(mb, h, w, dim).permute(0,3,1,2)

            for ind in range(mb_start_idx, mb_end_idx):
                if ind == 0:
                    feature_field_ensembled = feature_field[0:1,:,:,:]
                    continue

                i, j = ind_i_j[ind]
                h_mag = abs(i)
                w_mag = abs(j)

                feature_field_temp = feature_field[ind-mb_start_idx:ind-mb_start_idx+1, :, :, :]

                feature_field_ensembled += feature_field_temp * ((self.patch_size - h_mag) * (self.patch_size - w_mag) / patch_pixs)
            
                if i > 0:
                    feature_field_ensembled[:,:,:-1,:] += feature_field_temp[:,:,1:,:]*(h_mag*(self.patch_size-w_mag) / patch_pixs)
                elif i < 0:
                    feature_field_ensembled[:,:,1:,:] += feature_field_temp[:,:,:-1,:]*(h_mag*(self.patch_size-w_mag) / patch_pixs)

                if j > 0:
                    feature_field_ensembled[:,:,:,:-1] += feature_field_temp[:,:,:,1:]*((self.patch_size-h_mag)*w_mag / patch_pixs)
                elif j < 0:
                    feature_field_ensembled[:,:,:,1:] += feature_field_temp[:,:,:,:-1]*((self.patch_size-h_mag)*w_mag / patch_pixs)

                if i > 0 and j > 0:
                    feature_field_ensembled[:,:,:-1,:-1] += feature_field_temp[:,:,1:,1:]*(h_mag*w_mag / patch_pixs)  
                elif i > 0 and j < 0:
                    feature_field_ensembled[:,:,:-1,1:] += feature_field_temp[:,:,1:,:-1]*(h_mag*w_mag / patch_pixs)  
                elif i < 0 and j > 0:
                    feature_field_ensembled[:,:,1:,:-1] += feature_field_temp[:,:,:-1,1:]*(h_mag*w_mag / patch_pixs)  
                elif i < 0 and j < 0:
                    feature_field_ensembled[:,:,1:,1:] += feature_field_temp[:,:,:-1,:-1]*(h_mag*w_mag / patch_pixs)  

        out = feature_field_ensembled.squeeze().permute(1,2,0) / b

        dim = out.shape[-1]
        out = out.reshape(-1, dim)

        if return_h_w:
            return out, h, w
        return out

class EnsembledExtractorFine(FeatureExtractor):
    def __init__(self, patch_size, dx, dy, bs, scale_factor = 1):
        self.avg_pool = nn.AvgPool2d(patch_size//scale_factor)
        self.dx = dx
        self.dy = dy
        self.bs = bs
        self.scale_factor = scale_factor
        super().__init__(patch_size)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        frame = frame.unsqueeze(0).cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dx, self.dx + 1):
            for j in range(- self.dy, self.dy + 1):
                aug_list.append(transforms.functional.affine(frame, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b, _, h, w = image_batch.shape
        h, w = int(h / self.patch_size), int(w / self.patch_size)

        feature_field_full = None
        ind = 1
        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            out = model.get_intermediate_layers(image_batch[mb_start_idx:mb_end_idx], n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token

            mb = out.shape[0]

            dim = out.shape[-1]
            feature_field = out.reshape(mb, h, w, dim).permute(0,3,1,2)
            if feature_field_full is None:
                feature_field_full = torch.nn.functional.interpolate(feature_field[:1], scale_factor=self.patch_size, mode='nearest')

            for ind in range(mb_start_idx, mb_end_idx):
                if ind == 0:
                    continue
                i, j = ind_i_j[ind]
                new_feature_field = torch.nn.functional.interpolate(feature_field[ind-mb_start_idx:ind-mb_start_idx+1, :, :, :], scale_factor=self.patch_size, mode='nearest')
                feature_field_full += transforms.functional.affine(new_feature_field, translate=[-i,-j], angle=0, scale=1, shear=0)

        feature_fine_mean = feature_field_full / b
        out = self.avg_pool(feature_fine_mean).squeeze().permute(1,2,0)

        dim = out.shape[-1]
        out = out.reshape(-1, dim)

        if return_h_w:
            return out, h * self.scale_factor, w * self.scale_factor
        return out


class DinoV2DepthBasicExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__(None)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        with torch.inference_mode():
            out = model.whole_inference(frame, img_meta=None, rescale=True)
        return out

class DinoV2DepthEnsembledExtractor(FeatureExtractor):
    def __init__(self, dx, dy, bs):
        self.dx = dx
        self.dy = dy
        self.bs = bs
        super().__init__(None)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        frame = frame.unsqueeze(0).cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dx, self.dx + 1):
            for j in range(- self.dy, self.dy + 1):
                aug_list.append(transforms.functional.affine(frame, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b, _, _, _ = image_batch.shape

        feature_field_full = None
        ind = 1
        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            with torch.inference_mode():
                feature_field = model.whole_inference(image_batch[mb_start_idx:mb_end_idx], img_meta=None, rescale=True)
            if feature_field_full is None:
                feature_field_full = feature_field[:1] #torch.nn.functional.interpolate(feature_field[:1], scale_factor=self.patch_size, mode='nearest')

            for ind in range(mb_start_idx, mb_end_idx):
                if ind == 0:
                    continue
                i, j = ind_i_j[ind]
                feature_field_full += transforms.functional.affine(feature_field[ind-mb_start_idx:ind-mb_start_idx+1, :, :, :], translate=[-i,-j], angle=0, scale=1, shear=0)

        feature_fine_mean = feature_field_full / b
        out = feature_fine_mean

        return out


class DinoV2DepthFeatsEnsembledExtractor(FeatureExtractor):
    def __init__(self, patch_size, dx, dy, bs):
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.dx = dx
        self.dy = dy
        self.bs = bs
        super().__init__(patch_size)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        img_size = frame.shape[1:]
        frame = frame.unsqueeze(0).cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dx, self.dx + 1):
            for j in range(- self.dy, self.dy + 1):
                aug_list.append(transforms.functional.affine(frame, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b = image_batch.shape[0]

        ind = 1

        feature_field_full_arrays = None

        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            out = model.extract_feat(image_batch[mb_start_idx:mb_end_idx])

            n_features = len(out)
            for feature_i in range(n_features):
                assert len(out[feature_i]) == 2
            if feature_field_full_arrays is None:
                feature_field_full_arrays = [[None, None] for _ in range(n_features)]

            # Only handle linear head and linear4/DPT head
            assert n_features == 1 or n_features == 4

            mb = out[0][0].shape[0]
            dim = out[0][0].shape[1]
            # feature_field = out.reshape(mb, h, w, dim).permute(0,3,1,2)

            if feature_field_full_arrays[0][0] is None:
                for feat_i in range(len(out)):
                    for feat_j in range(len(out[feat_i])):
                        feature_field = out[feat_i][feat_j]
                        if len(feature_field.shape) == 2:
                            feature_field_full_arrays[feat_i][feat_j] = feature_field[:1]
                        else:
                            feature_field_full_arrays[feat_i][feat_j] = torch.nn.functional.interpolate(feature_field[:1], scale_factor=self.patch_size, mode='nearest')


            for ind in range(mb_start_idx, mb_end_idx):
                if ind == 0:
                    continue
                i, j = ind_i_j[ind]
                for feat_i in range(len(out)):
                    for feat_j in range(len(out[feat_i])):
                        feature_field = out[feat_i][feat_j]
                        if len(feature_field.shape) == 2:
                            new_feature_field = feature_field[ind-mb_start_idx:ind-mb_start_idx+1]
                            feature_field_full_arrays[feat_i][feat_j] += new_feature_field
                        else:
                            new_feature_field = torch.nn.functional.interpolate(feature_field[ind-mb_start_idx:ind-mb_start_idx+1, :, :, :], scale_factor=self.patch_size, mode='nearest')
                            feature_field_full_arrays[feat_i][feat_j] += transforms.functional.affine(new_feature_field, translate=[-i,-j], angle=0, scale=1, shear=0)

        for feat_i in range(len(feature_field_full_arrays)):
            for feat_j in range(len(feature_field_full_arrays[feat_i])):
                feature_field_full_arrays[feat_i][feat_j] /= b
                if len(feature_field_full_arrays[feat_i][feat_j].shape) != 2:
                    feature_field_full_arrays[feat_i][feat_j] = self.avg_pool(feature_field_full_arrays[feat_i][feat_j])

        # Retain original tuple of tuple format
        out = tuple(tuple(f) for f in feature_field_full_arrays)

        # Get depth
        out = model._decode_head_forward_test(out, None)
        out = torch.clamp(out, min=model.decode_head.min_depth, max=model.decode_head.max_depth)
        out = torch.nn.functional.interpolate(input=out, size=img_size, scale_factor=None, mode="bilinear", align_corners=model.align_corners)

        return out


class DinoV2SegFeatsEnsembledExtractor(FeatureExtractor):
    def __init__(self, patch_size, dx, dy, bs):
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.dx = dx
        self.dy = dy
        self.bs = bs
        super().__init__(patch_size)

    def extract_feature(self, model, frame, return_h_w=False, img_meta=None):
        """Extract dense features."""
        img_size = frame.shape[1:]
        frame = frame.cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dx, self.dx + 1):
            for j in range(- self.dy, self.dy + 1):
                aug_list.append(transforms.functional.affine(frame, translate=[i,j], angle=0, scale=1, shear=0))
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b = image_batch.shape[0]
        # print(ind_i_j)
        ind = 1

        feature_field_full_arrays = None

        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            curr_batch_size = mb_end_idx - mb_start_idx
            out = model.extract_feat(image_batch[mb_start_idx:mb_end_idx])
            # print(len(out), len(out[0]), len(out[1]), len(out[2]), len(out[3]), out[0][1].shape)
            # for i in range(curr_batch_size):
            #     print(out[0][i].shape)

            n_features = len(out)
            for feature_i in range(n_features):
                assert len(out[feature_i]) == curr_batch_size

            # Only handle linear head 
            assert n_features == 4
            if feature_field_full_arrays is None:
                feature_field_full_arrays = [[] for _ in range(n_features)]

            mb = out[0][0].shape[0]
            dim = out[0][0].shape[1]
            # feature_field = out.reshape(mb, h, w, dim).permute(0,3,1,2)

            for feat_i in range(len(out)):
                for feat_j in range(len(out[feat_i])):
                    if mb_start_idx + feat_j == 0:
                        feature_field = out[feat_i][feat_j].unsqueeze(0)
                        feature_field_full_arrays[feat_i] = (torch.nn.functional.interpolate(feature_field, scale_factor=self.patch_size, mode='nearest'))
                        continue
                    i, j = ind_i_j[mb_start_idx + feat_j]
                    feature_field = out[feat_i][feat_j].unsqueeze(0)
                    feature_field = torch.nn.functional.interpolate(feature_field, scale_factor=self.patch_size, mode='nearest')
                    feature_field_full_arrays[feat_i] += (transforms.functional.affine(feature_field, translate=[-i,-j], angle=0, scale=1, shear=0))
        
        for feat_i in range(len(feature_field_full_arrays)):
            feature_field_full_arrays[feat_i] /= b
            # print(feature_field_full_arrays[feat_i].shape)
            feature_field_full_arrays[feat_i] = self.avg_pool(feature_field_full_arrays[feat_i])
        # print(len(feature_field_full_arrays), feature_field_full_arrays[0].shape)
        # Retain original tuple of tuple format
        out = tuple(f for f in feature_field_full_arrays)
        # print(len(out), len(out[0]), len(out[1]), len(out[2]), len(out[3]), out[0][0].shape)

        # Get depth
        out = model.decode_head.forward_test(out, img_meta, {'mode': 'slide', 'crop_size': (512, 512), 'stride': (341, 341)})
        out = resize(
            input=out,
            size=frame.shape[2:],
            mode='bilinear',
            align_corners=False)
        # out = torch.clamp(out, min=model.decode_head.min_depth, max=model.decode_head.max_depth)
        # out = torch.nn.functional.interpolate(input=out, size=img_size, scale_factor=None, mode="bilinear", align_corners=model.align_corners)
        return out

class SanityExtractor(FeatureExtractor):
    def __init__(self, patch_size, dataset):
        self.dataset = dataset
        self.sanity = True
        super().__init__(patch_size)


    def extract_feature(self, model, frame, frame_path, return_h_w=False):
        """Extract one frame feature everytime."""
        #out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
        #out = out[:, 1:, :]  # we discard the [CLS] token
        found = False
        for i in range(len(self.dataset.video_frame_list)):
            if self.dataset.video_frame_list[i] == frame_path:
                out = self.dataset[i][1].unsqueeze(0).cuda()
                found = True
                break
            else:
                continue
        if not found:
            raise ValueError("not found")

        h, w = int(frame.shape[1] / self.patch_size), int(frame.shape[2] / self.patch_size)
        dim = out.shape[-1]
        out = out[0].reshape(h, w, dim)
        out = out.reshape(-1, dim)
        if return_h_w:
            return out, h, w
        return out

class DinoV2DepthFeatsInterpolationExtractor(FeatureExtractor):
    def __init__(self, patch_size, dx, dy, bs):
        self.avg_pool = nn.AvgPool2d(patch_size)
        self.dx = dx
        self.dy = dy
        self.bs = bs
        self.patch_size = patch_size
        super().__init__(patch_size)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        img_size = frame.shape[1:]
        frame = frame.unsqueeze(0).cuda()

        out = model.extract_feat(frame) # b*c*h/p*w/p
        # print(out[0][0].shape, len(out), len(out[0]))
        feature_field = [[None for _ in range(len(out[0]))]] * len(out)
        for i in range(len(out)):
            for j in range(len(out[i])):
                if len(out[i][j].shape) == 2:
                    feature_field[i][j] = out[i][j]
                else:
                    temp = torch.nn.functional.interpolate(out[i][j], scale_factor=self.patch_size, mode='bicubic')
                    feature_field[i][j] = torch.nn.functional.avg_pool2d(temp, kernel_size=self.patch_size)
        
        # Get depth
        out = model._decode_head_forward_test(feature_field, None)
        out = torch.clamp(out, min=model.decode_head.min_depth, max=model.decode_head.max_depth)
        out = torch.nn.functional.interpolate(input=out, size=img_size, scale_factor=None, mode="bilinear", align_corners=model.align_corners)

        return out

class DinoV2DepthGaussianExtractor(FeatureExtractor):
    def __init__(self, dx, dy, bs):
        self.dx = dx
        self.dy = dy
        self.bs = bs
        super().__init__(None)

    def extract_feature(self, model, frame, return_h_w=False):
        """Extract dense features."""
        frame = frame.unsqueeze(0).cuda()
        aug_list = []
        aug_list.append(frame)
        # store dictionary
        ind_i_j = {}
        ind = 1
        for i in range(- self.dx, self.dx + 1):
            for j in range(- self.dy, self.dy + 1):
                gaussian_noise = torch.randn_like(frame) / 100  # guassian noise mean 0 variance 0.01
                aug_list.append(frame + gaussian_noise)
                ind_i_j[ind] = (i,j)
                ind += 1

        image_batch = torch.cat(aug_list, dim=0)
        b, _, _, _ = image_batch.shape

        feature_field_full = None
        ind = 1
        for mb_start_idx in range(0, b, self.bs):
            mb_end_idx = mb_start_idx + self.bs if mb_start_idx + self.bs < b else b
            with torch.inference_mode():
                feature_field = model.whole_inference(image_batch[mb_start_idx:mb_end_idx], img_meta=None, rescale=True)
            if feature_field_full is None:
                feature_field_full = feature_field[:1] #torch.nn.functional.interpolate(feature_field[:1], scale_factor=self.patch_size, mode='nearest')

            for ind in range(mb_start_idx, mb_end_idx):
                if ind == 0:
                    continue
                i, j = ind_i_j[ind]
                feature_field_full += feature_field[ind-mb_start_idx:ind-mb_start_idx+1, :, :, :]

        feature_fine_mean = feature_field_full / b
        out = feature_fine_mean

        return out