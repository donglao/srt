import os, glob
import numpy as np
from torchvision import transforms
from PIL import Image
import tqdm
import torch

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def compute_error_metrics(ground_truth, output):
    '''
    Computation of error metrics between predicted and ground truth depths
    '''

    thresh = np.maximum((ground_truth / output), (output / ground_truth))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (ground_truth - output) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(ground_truth) - np.log(output)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(ground_truth - output) / ground_truth)

    sq_rel = np.mean(((ground_truth - output) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class NYUv2TestingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) ground truth depth map (optional)

    Arg(s):
        dataset_path : list[str]
            paths to NYUv2 testing directory
    '''

    def load_image(self, path, resize_factor=1):
        '''
        Loads an image as Pillow image

        Arg(s):
            path : str
                path to 16-bit PNG file
            resize_factor : float

        Returns:
            Image : Pillow image
        '''

        image = Image.open(path).convert('RGB')
        image = image.resize((resize_factor * image.width, resize_factor * image.height))

        return image

    def load_depth(self, path, multiplier=256.0, data_format='HW'):
        '''
        Loads a depth map from a 16-bit PNG file

        Arg(s):
            path : str
                path to 16-bit PNG file
            multiplier : float
                multiplier for encoding float as 16/32 bit unsigned integer
            data_format : str
                HW, CHW, HWC
        Returns:
            numpy[float32] : depth map
        '''

        # Loads depth map from 16-bit PNG file
        z = np.array(Image.open(path), dtype=np.float32)

        # Assert 16-bit (not 8-bit) depth map
        z = z / multiplier
        z[z <= 0] = 0.0

        # Expand dimensions based on output format
        if data_format == 'HW':
            pass
        elif data_format == 'CHW':
            z = np.expand_dims(z, axis=0)
        elif data_format == 'HWC':
            z = np.expand_dims(z, axis=-1)
        else:
            raise ValueError('Unsupported data format: {}'.format(data_format))

        return z

    def __init__(self, dataset_dirpath, resize_factor=1):

        assert os.path.exists(dataset_dirpath), dataset_dirpath

        self.image_paths = sorted(glob.glob(
            os.path.join(dataset_dirpath, 'images', '*.png')))

        self.ground_truth_paths = sorted(
            glob.glob(os.path.join(dataset_dirpath, 'depths', '*.png')))
        
        assert len(self.image_paths) == len(self.ground_truth_paths)

        self.n_sample = len(self.image_paths)

        self.resize_factor = resize_factor
        self.transform = make_depth_transform()

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = self.load_image(self.image_paths[index])
        image = self.transform(image)

        # Load depth
        depth = self.load_depth(self.ground_truth_paths[index])

        return image, depth

    def __len__(self):
        return self.n_sample


def eval_nyu_v2(args, model, extractor=None):
    '''
    Reads dataset
    '''
    nyu_v2_testing_dataset_dirpath = os.path.join(args.data_path, 'testing')

    test_dataloader = torch.utils.data.DataLoader(
        NYUv2TestingDataset(
            dataset_dirpath=nyu_v2_testing_dataset_dirpath,
            resize_factor=1),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False)

    '''
    Define error metrics
    '''
    abs_rel_errors = np.zeros(test_dataloader.dataset.n_sample)
    sq_rel_errors = np.zeros(test_dataloader.dataset.n_sample)
    rmse_errors = np.zeros(test_dataloader.dataset.n_sample)
    rmse_log_errors = np.zeros(test_dataloader.dataset.n_sample)
    a1_accuracies = np.zeros(test_dataloader.dataset.n_sample)
    a2_accuracies = np.zeros(test_dataloader.dataset.n_sample)
    a3_accuracies = np.zeros(test_dataloader.dataset.n_sample)

    '''
    Inference and evaluate on ground truth
    '''
    for idx, (image, ground_truth) in enumerate(tqdm.tqdm(test_dataloader)):

        image = image.to("cuda")
        image = image.squeeze(0)
        with torch.inference_mode():
            pass
            if extractor is not None:
                output = extractor.extract_feature(model, image)
            else:
                output = model.whole_inference(image, img_meta=None, rescale=True)
        
        # Normalize output
        min_value, max_value = output.min(), output.max()
        normalized_values = (output - min_value) / (max_value - min_value)

        # Convert from normalized inverse depth to depth
        # depth = 1.0 / output
        depth = np.squeeze(output.cpu().numpy())
        
        # Perform median scaling
        ground_truth = np.squeeze(ground_truth.cpu().numpy())
        mask = np.where(ground_truth > 0, 1, 0)
        ground_truth = ground_truth[mask]
        depth = depth[mask]

        scale = np.median(ground_truth) / np.median(depth)
        depth = depth * scale

        # Evaluate depth
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_error_metrics(ground_truth, depth)
        # depth_image = render_depth(depth)
        # display(depth_image)
        # depth_image = render_depth(ground_truth)
        # display(depth_image)
        abs_rel_errors[idx] = abs_rel
        sq_rel_errors[idx] = sq_rel
        rmse_errors[idx] = rmse
        rmse_log_errors[idx] = rmse_log
        a1_accuracies[idx] = a1
        a2_accuracies[idx] = a2
        a3_accuracies[idx] = a3

    # Take mean over evaluation scores
    abs_rel_mean = np.mean(abs_rel_errors)
    sq_rel_mean = np.mean(sq_rel_errors)
    rmse_mean = np.mean(rmse_errors)
    rmse_log_mean = np.mean(rmse_log_errors)
    a1_mean = np.mean(a1_accuracies)
    a2_mean = np.mean(a2_accuracies)
    a3_mean = np.mean(a3_accuracies)
    print(abs_rel_errors)
    # Print scores
    print('Evaluation results:')
    print('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'AbsRel', 'SqRel', 'RMSE', 'RMSE_log', 'a1', 'a2', 'a3'))
    print('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        abs_rel_mean, sq_rel_mean, rmse_mean, rmse_log_mean, a1_mean, a2_mean, a3_mean))