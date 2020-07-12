DATASETS_SMALL = ['mnist_fashion','cifar10','cifar100']
DATASETS_BIG = ['food101N', 'clothing1M', 'clothing1M50k', 'clothing1Mbalanced']
DATASETS = DATASETS_SMALL + DATASETS_BIG

IMG_RESIZED = 256
IMG_CROPPED = 224

import numpy as np
import os, sys
import random
import pathlib

NUM_CLASSES = {'mnist_fashion':10, 'cifar10':10,  'cifar100':100, 'clothing1M':14, 'clothing1M50k':14, 'food101N':101}

try:
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms   
    from PIL import Image 
    class torch_dataset(Dataset): 
        def __init__(self, img_paths, labels, transform, num_classes): 
            self.transform = transform
            self.img_paths = img_paths
            self.labels = labels
            self.num_classes = num_classes
        def __getitem__(self, index):  
            img_path = self.img_paths[index]
            image = Image.open(img_path).convert('RGB')    
            img = self.transform(image)
            label = self.labels[img_path]
            return img, label
        def __len__(self):
            return len(self.img_paths)  
    class CustomTensorDataset(Dataset):
        def __init__(self, x_train, y_train, transform=None):
            self.x_train = x_train.copy()
            self.y_train = y_train.copy()
            self.transform = transform
            # transform for validation and test data
            if x_train.shape[1:] == (28,28):
                # no 3channel normalization for mnist
                self.transform2 = transforms.Compose([transforms.ToTensor()])
            else:
                self.transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
               
        def __getitem__(self, index):
            x = self.x_train[index]
            x = Image.fromarray(x)
            if self.transform:
                x = self.transform(x)
            else:
                x = self.transform2(x)
            y = int(self.y_train[index])
            return x, y
        def __len__(self):
            return self.x_train.shape[0]
except: 
    pass

def download_data(dataset_name):
    assert dataset_name in DATASETS, 'invalid dataset name!'
    
    import shutil
    import pickle
    import urllib
    try:
        from urllib.error import URLError
        from urllib.request import urlretrieve
    except ImportError:
        from urllib2 import URLError
        from urllib import urlretrieve
    def report_download_progress(chunk_number, chunk_size, file_size):
        if file_size != -1:
            percent = min(1, (chunk_number * chunk_size) / file_size)
            bar = '#' * int(64 * percent)
            sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))
    def download(destination_path, urlbase, resources):
        for resource in resources:
            path = os.path.join(os.getcwd(),'{}{}'.format(destination_path,resource))
            url = '{}{}'.format(urlbase,resource)
            if not os.path.exists(path):
                print('Downloading {} ...'.format(url))
                try:
                    hook = report_download_progress
                    urlretrieve(url, path, reporthook=hook)
                except URLError:
                    raise RuntimeError('Error downloading resource!')

    MNIST_RESOURCES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    if dataset_name == 'mnist':
        download('mnist/dataset/', 'http://yann.lecun.com/exdb/mnist/', MNIST_RESOURCES)
    elif dataset_name == 'mnist_fashion': 
        download('mnist_fashion/dataset/', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', MNIST_RESOURCES)
    elif dataset_name == 'cifar10':
        download('cifar10/dataset/', 'https://www.cs.toronto.edu/~kriz/', ['cifar-10-python.tar.gz'])
        shutil.unpack_archive('cifar10/dataset/cifar-10-python.tar.gz','cifar10/dataset/')
    elif dataset_name == 'cifar100':
        download('cifar100/dataset/', 'https://www.cs.toronto.edu/~kriz/', ['cifar-100-python.tar.gz'])
        shutil.unpack_archive('cifar100/dataset/cifar-100-python.tar.gz','cifar100/dataset/')
    elif dataset_name == 'food101N':
        if not os.path.isdir('food101N/dataset/Food-101N_release'):
            download('food101N/dataset/', 'https://food101n.blob.core.windows.net/food101n/', ['Food-101N_release.zip'])
            shutil.unpack_archive('food101N/dataset/Food-101N_release.zip','food101N/dataset/')
            os.remove('food101N/dataset/Food-101N_release.zip')
    elif dataset_name == 'clothing1M' or dataset_name == 'clothing1M50k' or dataset_name == 'clothing1Mbalanced':
        assert os.path.exists('clothing1M/dataset/venn.png'), 'Download Clothing1M dataset to clothing1M/dataset!'
        if os.path.exists('clothing1M/dataset/annotations.zip'):
            shutil.unpack_archive('clothing1M/dataset/annotations.zip','clothing1M/dataset')
            os.remove('clothing1M/dataset/annotations.zip')
            for i in range(10):
                shutil.unpack_archive('clothing1M/dataset/images/{}.tar'.format(i),'clothing1M/dataset/images')
                os.remove('clothing1M/dataset/images/{}.tar'.format(i))

def add_noise(dataset_name, y_train, noise_type, noise_ratio, num_classes):
    from numpy.testing import assert_array_almost_equal
    
    def noise_featuredependent(y_train_int, probs, noise_ratio):
        from sklearn.utils.multiclass import unique_labels
        def get_sorted_idx(probs, labels, class_id=None):
            '''
            Returns indices of samples beloning to class_id. Indices are sorted according to probs. First one is least confidently class_id
            and last one is most confidently class_id.
            If class_id is None, then just sorts all samples according to given probability
            '''
            # indices of samples which belong to class i
            if class_id is None:
                idx_i = labels
            else:
                idx_i = np.where(labels == class_id)[0]
            # order according to probabilities of confidence. First one is least likely for class i and last one is most likely
            idx_tmp = np.argsort(probs[idx_i])
            idx_sorted = idx_i[idx_tmp]

            # make sure sorted idx indeed belongs to given class
            if class_id is not None:
                assert np.sum(labels[idx_sorted] == class_id) == len(idx_sorted)
            # make sure idx are indeed sorted
            assert np.sum(np.diff(probs[idx_sorted])<0) == 0

            return idx_sorted
        y_noisy = np.copy(y_train_int)
        num_classes = len(unique_labels(y_train_int))
        num_noisy = int(y_train_int.shape[0]*noise_ratio)
        # class ids sorted according to their probabilities for each instance shape=(num_samples,num_classes)
        prob_preds = np.argsort(probs, axis=1)
        # first and second predicted classes for each instance shape=(num_samples)
        prob_pred1, prob_pred2 = prob_preds[:,-1], prob_preds[:,-2]
        # indices of wrong predictions for first prediction 
        idx_wrong = np.where(prob_pred1 != y_train_int)[0]
        # change mis-predicted instances to their first prediction because it is most similar to that class
        if len(idx_wrong) >= num_noisy:
            # get the probabilities of first predictions for each sample shape=(num_samples)
            prob1 = np.array([probs[i,prob_pred1[i]] for i in range(len(prob_pred1))])
            # sorted list of second prediction probabilities
            idx_sorted = np.argsort(prob1)
            # sort them according to prob1
            idx_wrong2 = get_sorted_idx(prob1, idx_wrong)
            # get elements with highest probability on second prediction because they are closest to other classes
            idx2change = idx_wrong2[-num_noisy:]
            # change them to their most likely class which is second most probable prediction
            y_noisy[idx2change] = prob_pred1[idx2change]
        else:
            y_noisy[idx_wrong] = prob_pred1[idx_wrong]
            # remaining number of elements to be mislabeled
            num_noisy_remain = num_noisy - len(idx_wrong)
            # get the probabilities of second predictions for each sample shape=(num_samples)
            prob2 = np.array([probs[i,prob_pred2[i]] for i in range(len(prob_pred2))])
            # sorted list of second prediction probabilities
            idx_sorted = np.argsort(prob2)
            # remove already changed indices for wrong first prediction
            idx_wrong2 = np.setdiff1d(idx_sorted, idx_wrong)
            # sort them according to prob2
            idx_wrong2 = get_sorted_idx(prob2, idx_wrong2)
            # get elements with highest probability on second prediciton because they are closest to other classes
            idx2change = idx_wrong2[-num_noisy_remain:]
            # change them to their most likely class which is second most probable prediction
            y_noisy[idx2change] = prob_pred2[idx2change]
            # get indices where second prediction has zero probability
            idx_tmp = np.where(prob2[idx2change] == 0)[0]
            idx_prob0 = idx2change[idx_tmp]
            assert np.sum(prob2[idx_prob0] != 0) == 0
            # since there is no information in second prediction, to prevent all samples with zero probability on second prediction to have same class
            # we will choose a random class for that sample
            for i in idx_prob0:
                classes = np.arange(num_classes)
                classes_clipped = np.delete(classes, y_train_int[i])
                y_noisy[i] = np.random.choice(classes_clipped, 1)

        return y_noisy
    def cm_uniform(num_classes, noise_ratio):
        # if noise ratio is integer, convert it to float
        if noise_ratio > 1 and noise_ratio < 100:
            noise_ratio = noise_ratio / 100.
        assert (noise_ratio >= 0.) and (noise_ratio <= 1.)

        P = noise_ratio / (num_classes - 1) * np.ones((num_classes, num_classes))
        np.fill_diagonal(P, (1 - noise_ratio) * np.ones(num_classes))

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P
    def noise_cm(y, cm):
        assert_array_almost_equal(cm.sum(axis=1), 1, 1)
        y_noisy = np.copy(y)
        num_classes = cm.shape[0]

        for i in range(num_classes):
            # indices of samples belonging to class i
            idx = np.where(y == i)[0]
            # number of samples belonging to class i
            n_samples = len(idx)
            for j in range(num_classes):
                if i != j:
                    # number of noisy samples according to confusion matrix
                    n_noisy = int(n_samples*cm[i,j])
                    if n_noisy > 0:
                        # indices of noisy samples
                        noisy_idx = np.random.choice(len(idx), n_noisy, replace=False)
                        # change their classes
                        y_noisy[idx[noisy_idx]] = j
                        # update indices
                        idx = np.delete(idx, noisy_idx)
        return y_noisy

    if noise_ratio > 0:
        assert noise_type in ['feature-dependent', 'class-dependent', 'symmetric']
        if noise_type == 'feature-dependent':
            train_preds = np.load('{}/dataset/student_logits.npy'.format(dataset_name))
            y_train_noisy = noise_featuredependent(y_train, train_preds, noise_ratio)
        elif noise_type == 'symmetric':
            P = cm_uniform(num_classes, noise_ratio)
            y_train_noisy = noise_cm(y_train, P)
        elif noise_type == 'class-dependent':
            if noise_ratio > 1 and noise_ratio < 100:
                noise_ratio = noise_ratio / 100.
            P = np.zeros((num_classes, num_classes))
            np.fill_diagonal(P, np.ones(num_classes))

            P[2,2] = 1 - noise_ratio
            P[3,3] = 1 - noise_ratio
            P[4,4] = 1 - noise_ratio
            P[5,5] = 1 - noise_ratio
            P[9,9] = 1 - noise_ratio
            P[2,0] = noise_ratio
            P[3,5] = noise_ratio
            P[4,7] = noise_ratio
            P[5,3] = noise_ratio
            P[9,1] = noise_ratio

            y_train_noisy = noise_cm(y_train, P)
        print('Synthetic noise ratio is {}'.format(np.sum(y_train_noisy!=y_train)/y_train.shape[0]))
        return y_train_noisy
    else:
        return y_train

def get_cm(dataset_name, noise_type, noise_ratio, framework=None):
    framework = get_framework(framework)
    num_classes = NUM_CLASSES[dataset_name]
    if dataset_name in DATASETS_SMALL:
        data_clean, _, _, _ = get_dataloader(dataset_name,32,framework,noise_type,0,0)
        data_noisy, _, _, _ = get_dataloader(dataset_name,32,framework,noise_type,noise_ratio,0)
        P = np.zeros((num_classes,num_classes))
        for (_, l_noisy), (_, l_clean) in zip(data_noisy,data_clean):
            P[l_clean,l_noisy] +=1
        P /= P.sum(axis=1)
    elif dataset_name == 'food101N':
        def get_label(file_path):
            path = os.path.normpath(file_path)
            parts = path.split(os.sep)
            label = parts[-2] == class_names
            return int(np.argmax(label))
        data_dir = 'food101N/dataset/Food-101N_release/'
        image_dir = data_dir+'/images'
        class_names = np.array([item.name for item in pathlib.Path(image_dir).glob('*')])
        verified_train_paths = {}
        with open(data_dir+'meta/verified_train.tsv','r') as f:
            lines = f.read().splitlines()
        for l in lines[1:]:
            entry = l.split()           
            img_path = data_dir+'images/'+entry[0]
            verified_train_paths[img_path] = int(entry[1])
        P = np.zeros((num_classes,num_classes))
        for key in verified_train_paths:
            label = get_label(key)
            if verified_train_paths[key] == 1:
                P[label,label] +=1
            elif verified_train_paths[key] == 0:
                P[:,label] +=1
        P /= P.sum(axis=1)   
    elif dataset_name == 'clothing1M' or dataset_name == 'clothing1M50k':
        data_dir = 'clothing1M/dataset/'
        clean_labels, noisy_labels = {}, {}
        with open(data_dir+'clean_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = data_dir+entry[0]
            clean_labels[img_path] = int(entry[1]) 

        with open(data_dir+'noisy_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = data_dir+entry[0]
            noisy_labels[img_path] = int(entry[1])

        result_dict = {}
        for key in noisy_labels:
            if key in clean_labels:
                result_dict[key] = {}
                result_dict[key]['clean'] = clean_labels[key]
                result_dict[key]['noisy'] = noisy_labels[key]
        for key in clean_labels:
            if key in noisy_labels:
                if key not in result_dict:
                    result_dict[key] = {}
                    result_dict[key]['clean'] = clean_labels[key]
                    result_dict[key]['noisy'] = noisy_labels[key]
        P = np.zeros((num_classes,num_classes))
        for key in result_dict:
            P[result_dict[key]['clean'],result_dict[key]['noisy']] +=1
        P /= P.sum(axis=1)
    return P

def get_synthetic_idx(dataset_name,seed,num_metadata,num_validation,noise_type,noise_ratio):
    if dataset_name in DATASETS_SMALL:
        _, y_clean, _, _, _, _, _, _, _ = get_smalldata(dataset_name,seed,num_metadata,num_validation,noise_type, 0)
        _, y_noisy, _, _, _, _, _, _, _ = get_smalldata(dataset_name,seed,num_metadata,num_validation,noise_type, noise_ratio)
        return y_clean != y_noisy, y_clean
    return None,None

def get_smalldata(dataset_name, random_seed, num_metadata ,num_validation, noise_type='feature-dependent', noise_ratio=0):
    assert dataset_name in DATASETS_SMALL, 'invalid dataset name!'
    import gzip
    import pickle
    from sklearn.model_selection import train_test_split

    def load_mnist(path):
        """Load MNIST data from `path`"""
        train_labels_path = os.path.join(path,'train-labels-idx1-ubyte.gz')
        train_images_path = os.path.join(path,'train-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(path,'t10k-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(path,'t10k-images-idx3-ubyte.gz')

        with gzip.open(train_labels_path, 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(train_images_path, 'rb') as imgpath:
            x_train = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_train), 28, 28)
        with gzip.open(test_labels_path, 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(test_images_path, 'rb') as imgpath:
            x_test = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_test), 28, 28)
        return x_train, y_train, x_test, y_test
    def load_cifar10(data_dir, negatives=False):
        """
        Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
        """
        def unpickle(file):
            """load the cifar-10 data"""

            with open(file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            return data

        # get the meta_data_dict
        # num_cases_per_batch: 1000
        # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # num_vis: :3072

        meta_data_dict = unpickle(data_dir + "/batches.meta")
        cifar_label_names = meta_data_dict[b'label_names']
        cifar_label_names = np.array(cifar_label_names)

        # training data
        cifar_train_data = None
        cifar_train_filenames = []
        cifar_train_labels = []

        # cifar_train_data_dict
        # 'batch_label': 'training batch 5 of 5'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list

        for i in range(1, 6):
            cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
            if i == 1:
                cifar_train_data = cifar_train_data_dict[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
            cifar_train_filenames += cifar_train_data_dict[b'filenames']
            cifar_train_labels += cifar_train_data_dict[b'labels']

        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        if negatives:
            cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
        else:
            cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
        cifar_train_filenames = np.array(cifar_train_filenames)
        cifar_train_labels = np.array(cifar_train_labels)

        # test data
        # cifar_test_data_dict
        # 'batch_label': 'testing batch 1 of 1'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list

        cifar_test_data_dict = unpickle(data_dir + "/test_batch")
        cifar_test_data = cifar_test_data_dict[b'data']
        cifar_test_filenames = cifar_test_data_dict[b'filenames']
        cifar_test_labels = cifar_test_data_dict[b'labels']

        cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
        if negatives:
            cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
        else:
            cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
        cifar_test_filenames = np.array(cifar_test_filenames)
        cifar_test_labels = np.array(cifar_test_labels)

        #return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        #    cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
        return cifar_train_data, cifar_train_labels, cifar_test_data, cifar_test_labels
    def load_cifar100(data_dir):
        train_files = ['train']
        train_files = [os.path.join(data_dir, f) for f in train_files]
        test_files = ['test']
        test_files = [os.path.join(data_dir, f) for f in test_files]
        label_func = lambda x: np.array(x['fine_labels'], dtype='int32')

        # Load the data into memory
        def load_files(filenames):
            data = np.array([])
            labels = np.array([])
            for name in filenames:
                with open(name, 'rb') as f:
                    mydict = pickle.load(f, encoding='latin1')

                # The labels have different names in the two datasets.
                newlabels = label_func(mydict)
                if data.size:
                    data = np.vstack([data, mydict['data']])
                    labels = np.hstack([labels, newlabels])
                else:
                    data = mydict['data']
                    labels = newlabels
            data = np.reshape(data, [-1, 3, 32, 32], order='C')
            data = np.transpose(data, [0, 2, 3, 1])
            return data, labels

        train_data, train_labels = load_files(train_files)
        test_data, test_labels = load_files(test_files)

        return train_data, train_labels, test_data, test_labels

    if dataset_name == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist('{}/dataset/'.format(dataset_name))
        num_classes = 10
        class_names = np.arange(num_classes)
    elif dataset_name == 'mnist_fashion':
        x_train, y_train, x_test, y_test = load_mnist('{}/dataset/'.format(dataset_name))
        # add synthetic noise
        num_classes = 10
        class_names = ['Tshirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
    elif dataset_name == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10('cifar10/dataset/cifar-10-batches-py')        
        # add synthetic noise
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'cifar100':
        x_train, y_train, x_test, y_test = load_cifar100('cifar100/dataset/cifar-100-python')
        # add synthetic noise
        num_classes = 100
        class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                        'worm'] 

    # add synthetic noise
    y_train = add_noise(dataset_name, y_train, noise_type, noise_ratio, num_classes)
    # they are 2D originally in cifar
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    x_val, y_val, x_meta, y_meta = None, None, None, None
    x, x_test, y, y_test = train_test_split(x_test, y_test, test_size=5000, random_state=random_seed)
    if num_metadata > 0:
        x, x_meta, y, y_meta = train_test_split(x, y, test_size=num_metadata, random_state=random_seed)
    if num_validation > 0:
        if num_validation == x.shape[0]:
            x_val, y_val = x, y
        else:
            x, x_val, y, y_val = train_test_split(x, y, test_size=num_validation, random_state=random_seed)
    #if x.shape[0] > 0:
    #    x_train, y_train = np.concatenate((x_train, x)), np.concatenate((y_train, y))

    return x_train, y_train, x_val, y_val, x_test, y_test, x_meta, y_meta, class_names

def get_bigdata_lists(dataset_name,random_seed):
    if dataset_name == 'food101N':
        from sklearn.model_selection import train_test_split
        def get_label(file_path):
            path = os.path.normpath(file_path)
            parts = path.split(os.sep)
            label = parts[-2] == class_names
            return int(np.argmax(label))

        data_dir = 'food101N/dataset/Food-101N_release/'
        image_dir = data_dir+'/images'
        class_names = np.array([item.name for item in pathlib.Path(image_dir).glob('*')])
        img_paths = []
        verified_train_paths = {}
        verified_val_paths = {}
        with open(data_dir+'meta/imagelist.tsv','r') as f:
            lines = f.read().splitlines()
        for l in lines[1:]:
            img_path = data_dir+'images/'+l
            img_paths.append(img_path)
        with open(data_dir+'meta/verified_train.tsv','r') as f:
            lines = f.read().splitlines()
        for l in lines[1:]:
            entry = l.split()           
            img_path = data_dir+'images/'+entry[0]
            verified_train_paths[img_path] = int(entry[1])
        with open(data_dir+'meta/verified_val.tsv','r') as f:
            lines = f.read().splitlines()
        for l in lines[1:]:
            entry = l.split()           
            img_path = data_dir+'images/'+entry[0]
            verified_val_paths[img_path] = int(entry[1])

        train_imgs = []
        train_labels = {}
        val_imgs = []
        val_labels = {}
        test_imgs = []
        test_labels = {}

        val_imgs_tmp = []
        val_labels_tmp = []
        for key in verified_train_paths:
            val_imgs_tmp.append(key)
            val_labels_tmp.append(get_label(key))
        _, val_imgs, _, val_labels_tmp = train_test_split(val_imgs_tmp, val_labels_tmp, test_size=10000, random_state=random_seed)
        for key in val_imgs:
            val_labels[key] = get_label(key)

        # take only verified samples for test set
        for key in img_paths:
            if key in verified_val_paths:
                if verified_val_paths[key] == 1:
                    test_imgs.append(key)
                else:
                    train_imgs.append(key)
            elif key not in val_imgs:
                train_imgs.append(key)

        for key in train_imgs:
            train_labels[key] = get_label(key)
        for key in val_imgs:
            val_labels[key] = get_label(key)
        for key in test_imgs:
            test_labels[key] = get_label(key)

        random.Random(random_seed).shuffle(train_imgs)
        return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, val_imgs, val_labels, class_names
    elif dataset_name == 'clothing1M' or dataset_name == 'clothing1M50k' or dataset_name == 'clothing1Mbalanced':
        data_dir = 'clothing1M/dataset/'
        class_names = ['T-Shirt','Shirt','Knitwear','Chiffon','Sweater','Hoodie','Windbreaker','Jacket','Downcoat','Suit','Shawl','Dress','Vest','Underwear']
        num_classes = len(class_names)
        train_imgs = []
        test_imgs = []
        val_imgs = []
        train_labels = {}
        test_labels = {}
        with open(data_dir+'noisy_train_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = data_dir+l
            train_imgs.append(img_path)

        if dataset_name == 'clothing1M50k':
            with open(data_dir+'clean_train_key_list.txt','r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = data_dir+l
                train_imgs.append(img_path)

        with open(data_dir+'clean_test_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = data_dir+l
            test_imgs.append(img_path)

        with open(data_dir+'clean_val_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = data_dir+l
            val_imgs.append(img_path)
            
        with open(data_dir+'noisy_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = data_dir+entry[0]
            train_labels[img_path] = int(entry[1])

        with open(data_dir+'clean_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = data_dir+entry[0]
            test_labels[img_path] = int(entry[1]) 
            # correct train images labels with true labels if given
            if dataset_name == 'clothing1M50k':
                train_labels[img_path] = int(entry[1])
        random.Random(random_seed).shuffle(train_imgs)

        if dataset_name == 'clothing1Mbalanced':
            from collections import Counter
            # find class with minimum num of samples
            min_samples = 1000000
            for i in range(num_classes):
                num_samples = Counter(train_labels.values())[i]
                if num_samples < min_samples:
                    min_samples = num_samples
            # set all classes to equal number of samples = min_samples
            class_counts = np.zeros(num_classes)
            train_imgs2 = []
            train_labels2 = {}
            for img in train_imgs:
                label = train_labels[img]
                if class_counts[label] < min_samples:
                    train_imgs2.append(img)
                    train_labels2[img] = label
                    class_counts[label] += 1
            train_imgs = train_imgs2
            train_labels = train_labels2
        random.Random(random_seed).shuffle(train_imgs)

        return train_imgs, train_labels, val_imgs, test_labels, test_imgs, test_labels, val_imgs, test_labels, class_names 

def get_bigdata_tf(dataset_name,random_seed):
    import tensorflow as tf
    import pathlib

    def get_procfunc(labels):
        def process_func(img_path):
            # read image
            img = tf.io.read_file(img_path)
            # convert the compressed string to a 3D uint8 tensor
            img = tf.image.decode_jpeg(img, channels=3)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            # resize the image to the desired size.
            img = tf.image.resize(img, [IMG_RESIZED, IMG_RESIZED])
            # crop from center to resize 224
            img = tf.image.central_crop(img, IMG_CROPPED/IMG_RESIZED)
            # normalization of mean and std
            img = tf.image.per_image_standardization(img)
            label = tf.cast(labels[img_path.decode('utf-8')], tf.int32)
            return img, label
        return process_func
    def set_shapes(img, label, img_shape):
        img.set_shape(img_shape)
        label.set_shape([])
        return img, label

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, meta_imgs, meta_labels, class_names = get_bigdata_lists(dataset_name,random_seed)

    train_ds = tf.data.Dataset.from_tensor_slices(train_imgs).map(lambda x: tf.numpy_function(func=get_procfunc(train_labels), inp=[x], Tout=[tf.float32, tf.int32]))
    test_ds = tf.data.Dataset.from_tensor_slices(test_imgs).map(lambda x: tf.numpy_function(func=get_procfunc(test_labels), inp=[x], Tout=[tf.float32, tf.int32]))
    # this part is required because after mapping dataset with tf.numpy_function, it looses its shape
    img_shape = [IMG_CROPPED, IMG_CROPPED, 3]
    train_ds = train_ds.map(lambda img, label: set_shapes(img, label, img_shape))
    test_ds = test_ds.map(lambda img, label: set_shapes(img, label, img_shape))

    val_ds, meta_ds = None, None
    if not val_imgs:
        val_ds = tf.data.Dataset.from_tensor_slices(val_imgs).map(lambda x: tf.numpy_function(func=get_procfunc(val_labels), inp=[x], Tout=[tf.float32, tf.int32]))
        val_ds = val_ds.map(lambda img, label: set_shapes(img, label, img_shape))
    if not meta_imgs:
        meta_ds = meta_ds.map(lambda img, label: set_shapes(img, label, img_shape))
        meta_ds = tf.data.Dataset.from_tensor_slices(meta_imgs).map(lambda x: tf.numpy_function(func=get_procfunc(meta_labels), inp=[x], Tout=[tf.float32, tf.int32]))
    
    return train_ds, val_ds, test_ds, meta_ds, class_names

def get_bigdata_torch(dataset_name,random_seed):
    import torchvision.transforms as transforms      
    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, meta_imgs, meta_labels, class_names = get_bigdata_lists(dataset_name,random_seed)
    transform = transforms.Compose([
            transforms.Resize(IMG_RESIZED),
            transforms.CenterCrop(IMG_CROPPED),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ]) 
    val_ds, meta_ds = None, None
    train_ds = torch_dataset(train_imgs,train_labels,transform,len(class_names))
    test_ds = torch_dataset(test_imgs,test_labels,transform,len(class_names))
    val_ds = torch_dataset(val_imgs,val_labels,transform,len(class_names))
    meta_ds = torch_dataset(meta_imgs,meta_labels,transform,len(class_names))
    return train_ds, val_ds, test_ds, meta_ds, class_names

def get_framework(framework=None):
    # get the installed framework. either tensorflow or pytorch
    if framework == None:
        try:
            import tensorflow as tf
            framework = 'tensorflow'
        except ImportError:
            try:
                import torch
                framework = 'pytorch'
            except ImportError:
                raise ImportError('Either tensorflow or pytorch should be installed!')
    assert framework in ['tensorflow', 'pytorch'], 'Framework should be either tensorflow or pytorch!'
    return framework

def get_smalldata_transform_func(framework, dataset_name):
    if framework == 'pytorch':
        import torchvision.transforms as transforms
        if dataset_name == 'mnist_fashion':
            return transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ])
    if framework == 'tensorflow':
        import tensorflow as tf
        def transformfunc(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.pad_to_bounding_box(img, 3, 3, 38, 38)
            img = tf.image.random_crop(img, [32,32,3])
            img = tf.image.per_image_standardization(img)
            return img, label
        def transformfunc_mnist(img, label):
            img = tf.image.per_image_standardization(img)
            return img, label
        if dataset_name == 'mnist_fashion':
            return transformfunc_mnist
        else:
            return transformfunc

def get_data(dataset_name, framework=None, noise_type='feature-dependent', noise_ratio=0, random_seed=42, num_metadata=0,num_validation=5000):
    assert dataset_name in DATASETS, 'invalid dataset name!'

    framework = get_framework(framework)
    download_data(dataset_name)

    if dataset_name in DATASETS_SMALL:
        val_dataset, meta_dataset = None, None
        x_train, y_train, x_val, y_val, x_test, y_test,  x_meta, y_meta, class_names = get_smalldata(dataset_name,random_seed,num_metadata,num_validation,noise_type,noise_ratio)
        transorm_func = get_smalldata_transform_func(framework, dataset_name)
        if framework == 'tensorflow':
            import tensorflow as tf
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(transorm_func)
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            if not (x_val is None):
                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            if not (x_meta is None):
                meta_dataset = tf.data.Dataset.from_tensor_slices((x_meta, y_meta))
        elif framework == 'pytorch':
            train_dataset = CustomTensorDataset(x_train, y_train, transform=transorm_func)
            test_dataset = CustomTensorDataset(x_test, y_test, transform=None)
            if not (x_val is None):
                val_dataset = CustomTensorDataset(x_val, y_val, transform=None)
            if not (x_meta is None):
                meta_dataset = CustomTensorDataset(x_meta, y_meta, transform=None)
        return train_dataset, val_dataset, test_dataset, meta_dataset, class_names
    else:
        if framework == 'tensorflow':
            return get_bigdata_tf(dataset_name,random_seed)
        elif framework == 'pytorch':
            return get_bigdata_torch(dataset_name,random_seed)

def get_dataloader(dataset_name, batch_size, framework=None, noise_type='feature-dependent', noise_ratio=0, random_seed=42, num_workers=2):
    framework = get_framework(framework)
    train_dataset, val_dataset, test_dataset, _, class_names = get_data(dataset_name,framework,noise_type,noise_ratio,random_seed)
    if framework == 'tensorflow':
        train_dataloader = train_dataset.batch(batch_size)
        test_dataloader = test_dataset.batch(batch_size)
        val_dataloader = val_dataset.batch(batch_size) if val_dataset != None else None
    elif framework == 'pytorch':
        import torch
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False, num_workers=num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False, drop_last=True) if val_dataset != None else None
    return train_dataloader, val_dataloader, test_dataloader, class_names

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, meta_dataset, class_names = get_data('food101N',num_metadata=5000,num_validation=0, noise_type='class-dependent', noise_ratio=30, )
    print('train', len(train_dataset))
    print('test', len(test_dataset))
    if not (val_dataset is None):
        print('val', len(val_dataset))
    if not (meta_dataset is None):
        print('meta', len(meta_dataset))
