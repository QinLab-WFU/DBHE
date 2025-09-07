import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="vit-s", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=900, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="adamw", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--parallel-val", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=32, help="length of hashing binary")

    # special settings
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="number of blocks in transformer to freeze, None - freeze nothing, 0 - freeze only patch_embed",
    )
    parser.add_argument("--hyp-c", type=float, default=0.1, help="hyperbolic c, '0' enables sphere mode")
    parser.add_argument("--clip-r", type=float, default=2.3, help="feature clipping radius")
    parser.add_argument("--tau", type=float, default=0.2, help="cross-entropy temperature")
    args = parser.parse_args()

    # mods
    # args.backbone = "resnet50"
    # args.lr = 1e-4
    args.batch_size = 128
    # args.hyp_c = 0
    # args.device = "cuda:1"
    # args.parallel_val = False
    return args
