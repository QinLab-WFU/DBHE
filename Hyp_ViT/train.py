import json
import os
import time

import PIL
import torch
import torchvision.transforms as T
from loguru import logger
from timm.utils import AverageMeter

from _data import build_loader, get_topk, get_class_num
from _utils import (
    build_optimizer,
    calc_learnable_params,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
)
from config import get_config
from loss import NCALoss
from network import build_model


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["loss"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)
        z = net(images, False)

        loss = criterion(z, labels)
        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
        optimizer.step()

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args, True)

    # setup criterion
    criterion = NCALoss(args)

    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                parallel_val=args.parallel_val,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def prepare_loaders(args, bl_fnc):
    if args.backbone.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tr = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    eval_tr = T.Compose(
        [
            T.Resize(224, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )

    train_loader, query_loader, dbase_loader = (
        bl_fnc(
            args.data_dir,
            args.dataset,
            "train",
            train_tr,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_fnc(
            args.data_dir,
            args.dataset,
            "query",
            eval_tr,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
        bl_fnc(
            args.data_dir,
            args.dataset,
            "dbase",
            eval_tr,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
    )
    return train_loader, query_loader, dbase_loader


def main():
    init()
    args = get_config()

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loader)

        # for hash_bit in [16, 32, 64, 128]:
        for hash_bit in [32]:
            print(f"processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pkl") for x in os.listdir(args.save_dir)):
                # raise Exception(f"*.pkl exists in {args.save_dir}")
                print(f"*.pkl exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    # for x in rst:
    #     print(
    #         f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
    #     )
    print_in_md(rst)


if __name__ == "__main__":
    main()
