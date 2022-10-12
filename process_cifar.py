import os
import sys
import numpy as np
import pickle

severities = [1, 2, 3, 4, 5]
corruptions = ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur", "brightness", "fog", \
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast", "jpeg_compression", "elastic_transform"]

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def reshape(data):
    # original data shape: (N, 3072)
    # reshape the vector into height * weight * channel format
    nchw = data.reshape((data.shape[0], 3, 32, 32))
    nhwc = np.transpose(nchw, (0, 2, 3, 1))
    return nhwc

def get_dirnames(dataset):
    if dataset == "cifar-10c":
        cor_dirname = "CIFAR-10-C"
        org_dirname = "cifar-10-batches-py"
    if dataset == "cifar-100c":
        cor_dirname = "CIFAR-100-C"
        org_dirname = "cifar-100-python"
    return cor_dirname, org_dirname

def load_data(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d

def process_corrupted_data(cor_dir):
    '''
    process the downloaded data and save the results into the dedicated directories.
    for cifar-10-c or cifar-100-c, we parse data according to the severity label and save each in the corresponding directory.
    e.g., data of severity level 1 with corruption type 'fog' is saved as "./dataset/corrupted/severity-1/fog.npy"
    cor_dir: directory of corrupted dataset.
    '''
    os.chdir(cor_dir)

    label_all = np.load("labels.npy")

    print("python: processing data...")
    for i, corruption in enumerate(corruptions):
        corruption_file_name = corruption + ".npy"
        data_all = np.load(corruption_file_name)
        for severity in severities:
            data = data_all[(severity - 1) * 10000 : severity * 10000]
            label = label_all[(severity - 1) * 10000 : severity * 10000]
            new_data_dir = "./corrupted/severity-"+str(severity)+"/"+corruption+".npy"
            ensure_dir(new_data_dir)
            np.save(new_data_dir, data)

            if i == 0:
                new_label_dir = "./corrupted/severity-"+str(severity)+"/labels.npy"
                np.save(new_label_dir, label)

def process_original_data(org_dir, dataset):
    '''
    process the downloaded data and save the results into the dedicated directories.
    for cifar 10,
        we read all five batches of train data, concatenate them into a single dataset and save it to the "origin" directory.
        we read a test batch and save it in each severity level directories in 'corrupted' directory.
        e.g., "./dataset/corrupted/severity-1/test.npy"
    for cifar 100,
        we read a train data and save it to the "origin" directory.
        we read a test data and save it in each severity level directories in 'corrupted' directory.
        e.g., "./dataset/corrupted/severity-1/test.npy"
    org_dir: directory of origin dataset.
    dataset: dataset type, "cifar-10c" or "cifar-100c".
    '''
    os.chdir(org_dir)

    if dataset == "cifar-10c":

        # 1. load train data
        files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        data_list = []
        label_list = []

        for filename in files:
            data_dict = load_data(filename)
            batch_data = data_dict[b'data']
            batch_label = np.asarray(data_dict[b'labels'])
            reshaped_data = reshape(batch_data) # reshape data

            # append data and label to the predefined list
            data_list.append(reshaped_data)
            label_list.append(batch_label)

        # concat data and labels
        data = np.concatenate(np.asarray(data_list), axis=0)
        label = np.concatenate(np.asarray(label_list), axis=0)

        # 2. load test data
        test_file = "test_batch"
        test_data_dict = load_data(test_file)
        test_data = test_data_dict[b'data']
        test_data = reshape(test_data) # reshape data

    else:  # dataset == "cifar-100c"
        # 1. load train data
        data_dict = load_data("train")
        data = data_dict[b'data']
        label = np.asarray(data_dict[b'fine_labels'])
        data = reshape(data) # reshape data

        # 2. load test data
        test_data_dict = load_data("test")
        test_data = test_data_dict[b'data']
        test_data = reshape(test_data) # reshape data

    # save train data
    org_data_dir = f'{cor_dir}/origin'
    if not os.path.exists(org_data_dir):
        os.makedirs(org_data_dir)

    np.save(f"{org_data_dir}/original.npy", data)
    np.save(f"{org_data_dir}/labels.npy", label)

    # save test data in each severity directories in 'corrupted' directory
    for severity in severities:
        save_data_dir = f'{cor_dir}/corrupted/severity-{severity}/test.npy'
        ensure_dir(save_data_dir)
        np.save(save_data_dir, test_data)

    # for severity-all directory, match the data shape
    save_data_dir = f'{cor_dir}/corrupted/severity-all/test.npy'
    ensure_dir(save_data_dir)
    test_data_ = np.concatenate([test_data for i in range(len(severities))], axis=0)
    np.save(save_data_dir, test_data_)


if __name__=="__main__":
    # get directory names to save data
    dataset = sys.argv[1]
    cor_dirname, org_dirname = get_dirnames(dataset)

    # get full directories
    home_dir = os.getcwd()
    cor_dir = f'{home_dir}/dataset/{cor_dirname}'
    org_dir = f'{home_dir}/dataset/{org_dirname}'

    process_corrupted_data(cor_dir) # process corrupted data and save to the dedicated directory
    process_original_data(org_dir, dataset) # process original data and save to the dedicated directory




