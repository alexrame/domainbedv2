# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import pickle
import pandas as pd
import numpy as np

import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate


# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # correlation shift
    "CelebA_Blond",
    "Waterbirds"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    CHECKPOINT_FREQ = 100 ## DiWA ##
    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 50 ## DiWA ##
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100 ## DiWA ##
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500 ## DiWA ##
    N_STEPS = 15000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100 ## DiWA ##
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100 ## DiWA ##
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


# this class is adapted from https://github.com/chingyaoc/fair-mixup/blob/master/celeba/main_dp.py
class CelebA(torch.utils.data.Dataset):

    def __init__(self, dataframe, folder_dir, target_id, transform=None, cdiv=0, ccor=0):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.targets = np.concatenate(dataframe.labels.values).astype(int)
        gender_id = 20

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))

        u1 = len(nontarget_males) - int(
            (1 - ccor) * (len(nontarget_males) - len(nontarget_females))
        )
        u2 = len(target_females) - int((1 - ccor) * (len(target_females) - len(target_males)))
        selected_idx = nontarget_males[:u1] + nontarget_females + target_males + target_females[:u2]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))

        selected_idx = nontarget_males + nontarget_females[:int(
            len(nontarget_females) * (1 - cdiv)
        )] + target_males + target_females[:int(len(target_females) * (1 - cdiv))]
        self.targets = self.targets[selected_idx]
        self.file_names = self.file_names[selected_idx]

        target_idx0 = np.where(self.targets[:, target_id] == 0)[0]
        target_idx1 = np.where(self.targets[:, target_id] == 1)[0]
        gender_idx0 = np.where(self.targets[:, gender_id] == 0)[0]
        gender_idx1 = np.where(self.targets[:, gender_id] == 1)[0]
        nontarget_males = list(set(gender_idx1) & set(target_idx0))
        nontarget_females = list(set(gender_idx0) & set(target_idx0))
        target_males = list(set(gender_idx1) & set(target_idx1))
        target_females = list(set(gender_idx0) & set(target_idx1))
        print(len(nontarget_males), len(nontarget_females), len(target_males), len(target_females))

        self.targets = self.targets[:, self.target_id]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)
        return image, label


class WaterbirdsDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, transform):
        self.df = df
        self.transform = transform
        # import pdb; pdb.set_trace()

        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        # self.g = df["a"].tolist()

    # def __getitem__(self, i):
    #     return self.transform(self.x[i]), self.y[i]
    #     #, self.g[i]
    def __getitem__(self, i):
        image = self.transform(self.x[i])
        label = self.y[i]
        return image, label

    def __len__(self):
        return len(self.x)


class Waterbirds(MultipleDomainDataset):
    ENVIRONMENTS = ["att0", "att1", "balanced"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        environments = self.ENVIRONMENTS
        print(environments)

        self.input_shape = (
            3,
            224,
            224,
        )
        self.num_classes = 2  # seabird or not

        self.dir = os.path.join(root, "waterbirds/waterbird_complete95_forest2water2/")
        metadata = os.path.join(root, "metadata_waterbirds.csv")

        df = pd.read_csv(metadata)
        dftr = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["tr"])]
        dftr0 = dftr[dftr["a"] == 0]
        dftr1 = dftr[dftr["a"] == 1]

        dfte = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["te"])]

        transform = transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
                transforms.Resize((
                    int(224 * (256 / 224)),
                    int(224 * (256 / 224)),
                )),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.datasets = []
        for dfenv in [dftr0, dftr1, dfte]:
            self.datasets.append(WaterbirdsDataset(df=dfenv, root=self.dir, transform=transform))


class CelebA_Blond(MultipleDomainDataset):
    ENVIRONMENTS = ["unbalanced_1", "unbalanced_2", "balanced"]
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        environments = self.ENVIRONMENTS
        print(environments)

        self.input_shape = (3, 224, 224,)
        self.num_classes = 2 # blond or not

        dataframes = []
        for env_name in ('tr_env1', 'tr_env2', 'te_env'):
            with open(f'{root}/celeba/blond_split/{env_name}_df.pickle', 'rb') as handle:
                dataframes.append(pickle.load(handle))
        tr_env1, tr_env2, te_env = dataframes

        orig_w = 178
        orig_h = 218
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_path = f'{root}/celeba/img_align_celeba'
        transform = transforms.Compose([
            transforms.CenterCrop(min(orig_w, orig_h)),
            transforms.Resize(self.input_shape[1:]),
            transforms.ToTensor(),
            normalize,
        ])

        if hparams['data_augmentation']:
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_shape[1:],
                                             scale=(0.7, 1.0), ratio=(1.0, 1.3333333333333333)),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            if hparams.get('test_data_augmentation', False):
                transform = augment_transform
        else:
            augment_transform = transform

        cdiv = hparams.get('cdiv', 0)
        ccor = hparams.get('ccor', 1)

        target_id = 9
        tr_dataset_1 = CelebA(pd.DataFrame(tr_env1), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        tr_dataset_2 = CelebA(pd.DataFrame(tr_env2), images_path, target_id, transform=augment_transform,
                              cdiv=cdiv, ccor=ccor)
        te_dataset = CelebA(pd.DataFrame(te_env), images_path, target_id, transform=transform)

        self.datasets = [tr_dataset_1, tr_dataset_2, te_dataset]


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)
