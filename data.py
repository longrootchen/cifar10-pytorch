import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

cls_to_idx = {
    'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}


def get_train_transforms():
    normalize = transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201))
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def get_valid_transforms():
    normalize = transforms.Normalize(mean=(0.491, 0.482, 0.447), std=(0.202, 0.199, 0.201))
    return transforms.Compose([transforms.ToTensor(), normalize])


class CIFAR10Dataset(Dataset):

    def __init__(self, df, img_dir, phase='train'):
        assert phase in ('train', 'val')
        self.img_dir = img_dir
        self.df = df
        self.phase = phase

        if phase == 'train':
            self.transform = get_train_transforms()
        elif phase == 'val':
            self.transform = get_valid_transforms()

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.df.iloc[index]['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.df.iloc[index]['target']
        return image, label

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    import pandas as pd
    from torch.utils.data import DataLoader

    test_dir = os.path.join(os.curdir, 'datasets', 'test')
    test_df = pd.read_csv(os.path.join(os.curdir, 'datasets', 'test.csv'))
    ds = CIFAR10Dataset(test_df, test_dir, 'val')
    print(len(ds))

    dl = DataLoader(ds, batch_size=128, shuffle=False)
    for i, (input_, target) in enumerate(dl):
        print(i)
        print(input_.size())
        break
