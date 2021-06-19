class FashionDataset(Dataset):
    '''
    定义Dataset:
    - 用于加载训练和测试数据，请勿改动
    - 返回一张图片(3维Tensor)以及对应的标签(0-9)
    '''

    def __init__(self, datadir, transform, is_train=True):
        super().__init__()
        self.datadir = datadir
        self.img, self.label = self.load_data(self.datadir, is_train=is_train)
        self.len_data = len(self.img)
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.img[index]), self.label[index]

    def __len__(self):
        return self.len_data

    def load_data(self, datadir, is_train):
        dirname = os.path.join(datadir)
        files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

        paths = []
        for fname in files:
            paths.append(os.path.join(dirname, fname))
        if is_train:

            with gzip.open(paths[0], 'rb') as lbpath:
                label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open(paths[1], 'rb') as imgpath:
                img = np.frombuffer(imgpath.read(), np.uint8,
                                    offset=16).reshape(len(label), 28, 28, 1)
        else:
            with gzip.open(paths[2], 'rb') as lbpath:
                label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

            with gzip.open(paths[3], 'rb') as imgpath:
                img = np.frombuffer(imgpath.read(), np.uint8,
                                    offset=16).reshape(len(label), 28, 28, 1)
        return img, label