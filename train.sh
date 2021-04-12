python main_cifar.py --arch vgg11 --method float --lr 0.1 --save_model models/vgg11.float.pkl --abit 4 > vgg11.float.out
python main_cifar.py --arch res20 --method float --lr 0.1 --save_model models/res20.float.pkl --abit 4 > res20.float.out
python main_cifar.py --arch vgg11 --method bc --lr 0.01 --load_model models/vgg11.float.pkl --save_model models/vgg11.1b.pkl --abit 4 --wbit 1 --gd_alpha > vgg11.1b.out
python main_cifar.py --arch vgg11 --method bc --lr 0.01 --load_model models/vgg11.float.pkl --save_model models/vgg11.2b.pkl --abit 4 --wbit 2 --gd_alpha > vgg11.2b.out
python main_cifar.py --arch res20 --method bc --lr 0.01 --load_model models/res20.float.pkl --save_model models/res20.1b.pkl --abit 4 --wbit 1 --gd_alpha > res20.1b.out
python main_cifar.py --arch res20 --method bc --lr 0.01 --load_model models/res20.float.pkl --save_model models/res20.2b.pkl --abit 4 --wbit 2 --gd_alpha > res20.2b.out
