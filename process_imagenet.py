import os

### code for making imagenet validation data compatiable with ImageFolder!
root = './dataset/ImageNet-C/origin/Data/CLS-LOC/val/'
f = open(root + 'LOC_val_solution.csv', 'r')
i = 0
for l in f:
    if i == 0:  # ignore header
        i += 1
        continue
    filename = l.split(',')[0]
    label = l.split(',')[1].split(' ')[0]
    dir = root + label

    ### 1. make dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(os.path.join(root,filename,'.JPEG'))

    ### 2. move files to dir
    print(label)
    if os.path.isfile(os.path.join(root, filename + '.JPEG')):
        os.rename(os.path.join(root, filename + '.JPEG'), os.path.join(dir, filename + '.JPEG'))

    i += 1