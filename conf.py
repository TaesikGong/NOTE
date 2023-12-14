

args = None

CIFAR10Opt = {
    'name': 'cifar10',
    'batch_size': 128,

    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
                    "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}

CIFAR100Opt = {
    'name': 'cifar100',
    'batch_size': 128,

    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-100-C',
    'classes': ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                'bottles', 'bowls', 'cans', 'cups', 'plates',
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow',
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'num_class': 100,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["original",

                "test",

                "gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],
    'src_domains': ["original"],
    'tgt_domains': [
        "gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}


ImageNetOpt = {
    # referred to for hyperparams: https://github.com/Lornatang/ResNet-PyTorch/blob/9e529757ce0607aafeae2ddd97142201b3d4cadd/examples/imagenet/main.py
    'name': 'imagenet',
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/ImageNet-C',
    # 'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 1000,
    'severity': 5,
    # 'corruptions': ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur",
    #                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
    #                 "jpeg_compression", "elastic_transform"],
    'domains': ["gaussian_noise-1", "gaussian_noise-2", "gaussian_noise-3", "gaussian_noise-4", "gaussian_noise-5",
                "gaussian_noise-all",

                "shot_noise-1", "shot_noise-2", "shot_noise-3", "shot_noise-4", "shot_noise-5", "shot_noise-all",

                "impulse_noise-1", "impulse_noise-2", "impulse_noise-3", "impulse_noise-4", "impulse_noise-5",
                "impulse_noise-all",

                "defocus_blur-1", "defocus_blur-2", "defocus_blur-3", "defocus_blur-4", "defocus_blur-5",
                "defocus_blur-all",

                "glass_blur-1", "glass_blur-2", "glass_blur-3", "glass_blur-4", "glass_blur-5", "glass_blur-all",

                "motion_blur-1", "motion_blur-2", "motion_blur-3", "motion_blur-4", "motion_blur-5", "motion_blur-all",

                "zoom_blur-1", "zoom_blur-2", "zoom_blur-3", "zoom_blur-4", "zoom_blur-5", "zoom_blur-all",

                "snow-1", "snow-2", "snow-3", "snow-4", "snow-5", "snow-all",

                "frost-1", "frost-2", "frost-3", "frost-4", "frost-5", "frost-all",

                "fog-1", "fog-2", "fog-3", "fog-4", "fog-5", "fog-all",

                "brightness-1", "brightness-2", "brightness-3", "brightness-4", "brightness-5", "brightness-all",

                "contrast-1", "contrast-2", "contrast-3", "contrast-4", "contrast-5", "contrast-all",

                "elastic_transform-1", "elastic_transform-2", "elastic_transform-3", "elastic_transform-4",
                "elastic_transform-5", "elastic_transform-all",

                "pixelate-1", "pixelate-2", "pixelate-3", "pixelate-4", "pixelate-5", "pixelate-all",

                "jpeg_compression-1", "jpeg_compression-2", "jpeg_compression-3", "jpeg_compression-4",
                "jpeg_compression-5", "jpeg_compression-all",
                ],

    'src_domains': ["original"],
    'tgt_domains': ["gaussian_noise-5",
                    "shot_noise-5",
                    "impulse_noise-5",
                    "defocus_blur-5",
                    "glass_blur-5",
                    "motion_blur-5",
                    "zoom_blur-5",
                    "snow-5",
                    "frost-5",
                    "fog-5",
                    "brightness-5",
                    "contrast-5",
                    "elastic_transform-5",
                    "pixelate-5",
                    "jpeg_compression-5",

    ],
}
