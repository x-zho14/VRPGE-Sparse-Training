import os
import pathlib
import random
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy
import numpy as np
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_subnet,
    save_checkpoint,
    get_lr,
    fix_model_subnet,
)
from utils.schedulers import get_policy, assign_learning_rate
from args import args
import importlib
import data
import models
from utils.compute_flops import print_model_param_flops_sparse


def main():
    print(args)
    torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    try:
        main_worker(args)
    except KeyboardInterrupt as e:
        print("rep_count: ", args.rep_count)


def main_worker(args):
    args.finetuning = False
    args.stablizing = False
    train, validate, modifier = get_trainer(args)
    model = get_model(args)
    model = set_gpu(args, model)
    optimizer, weight_opt = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)
    criterion = nn.CrossEntropyLoss().cuda()
    args.gpu = args.multigpu[0]
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )
    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None
    # Save the initial state
    flops_reduction_list = []

    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        assign_learning_rate(weight_opt, 0.5 * (1 + np.cos(np.pi * epoch / args.epochs)) * args.weight_opt_lr)
        print("current lr: ", get_lr(optimizer))
        print("current weight lr: ", weight_opt.param_groups[0]["lr"])
        print("current prune rate: ", args.prune_rate)
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(data.train_loader, model, criterion, optimizer, epoch, args, writer=writer, weight_opt=weight_opt)
        train_time.update((time.time() - start_train) / 60)
        start_validation = time.time()
        acc1, acc5, losses = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)
        if epoch % 1 == 0 and args.conv_type != "DenseConv":
            print("=> compute model params and flops")
            c = 3
            input_res = 32
            flops_reduction = print_model_param_flops_sparse(model, input_res=input_res, multiply_adds=False, c=c)
            flops_reduction_list.append(flops_reduction.item())
            print(sum(flops_reduction_list)/len(flops_reduction_list))
            torch.cuda.empty_cache()
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
            save_checkpoint({"epoch": epoch + 1, "arch": args.arch, "state_dict": model.state_dict(), "best_acc1": best_acc1, "best_acc5": best_acc5, "best_train_acc1": best_train_acc1, "best_train_acc5": best_train_acc5, "optimizer": optimizer.state_dict(), "curr_acc1": acc1, "curr_acc5": acc5}, is_best, filename=ckpt_base_dir / f"epoch_{epoch}.state", save=save,)
        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(writer, prefix="diagnostics", global_step=epoch)
        end_epoch = time.time()
        print("best acc:%.2f, location:%d", best_acc1, log_base_dir)

    if args.finetune:
        best_acc1 = 0
        args.finetuning = True
        args.K = 1
        freeze_model_subnet(model)
        fix_model_subnet(model)
        args.batch_size = 128
        data = get_dataset(args)

        if args.sample_from_training_set:
            args.use_running_stats = False
            i = 0
            BESTACC1, BESTIDX = 0, 0
            BESTMODEL = None
            while i < 10:
                i += 1
                acc1, acc5, _ = validate(data.train_loader, model, criterion, args, None, epoch=args.start_epoch)
                if acc1 > BESTACC1:
                    BESTACC1 = acc1
                    BESTMODEL = copy.deepcopy(model)
                fix_model_subnet(model)
            model = copy.deepcopy(BESTMODEL)

        args.use_running_stats = True
        torch.cuda.empty_cache()
        args.lr = 0.001
        parameters = list(model.named_parameters())
        weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            weight_params,
            0.001,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=args.nesterov,
        )
        for epoch in range(0, 40):
            cur_lr = get_lr(optimizer)
            print("current lr: ", cur_lr)
            start_train = time.time()
            train_acc1, train_acc5 = train(
                data.train_loader, model, criterion, optimizer, epoch, args, writer=writer, weight_opt=None
            )
            train_time.update((time.time() - start_train) / 60)
            # evaluate on validation set
            start_validation = time.time()
            acc1, acc5, losses = validate(data.val_loader, model, criterion, args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
            best_train_acc1 = max(train_acc1, best_train_acc1)
            best_train_acc5 = max(train_acc5, best_train_acc5)
            save = ((epoch % args.save_every) == 0) and args.save_every > 0
            if is_best or save or epoch == args.epochs - 1:
                if is_best:
                    print(f"==> New best, saving at {ckpt_base_dir / 'model_best_finetune.pth'}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "best_acc5": best_acc5,
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        "optimizer": optimizer.state_dict(),
                        "curr_acc1": acc1,
                        "curr_acc5": acc5,
                    },
                    is_best,
                    filename=ckpt_base_dir / f"epoch_{epoch}.state",
                    save=save,
                )
            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )
            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()
            print("best acc:%.2f, location:%d", best_acc1, log_base_dir)

def get_trainer(args):
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    return trainer.train, trainer.validate, trainer.modifier

def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    print(f"=> Parallelizing on {args.multigpu} gpus")
    torch.cuda.set_device(args.multigpu[0])
    model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
        args.multigpu[0]
    )
    return model

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)
    return dataset

def get_model(args):
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print(model)
    return model

def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        if not args.train_weights_at_the_same_time:
            parameters = list(model.named_parameters())
            bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
            rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": bn_params,
                        "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                    },
                    {"params": rest_params, "weight_decay": args.weight_decay},
                ],
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        else:
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.SGD(
                score_params, lr=0.1, weight_decay=1e-6, momentum=0.9
            )
            optimizer2 = torch.optim.SGD(
                weight_params,
                args.weight_opt_lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=args.nesterov,
            )
            return optimizer1, optimizer2

    elif args.optimizer == "adam":
        if not args.train_weights_at_the_same_time:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.Adam(
                score_params, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer2 = torch.optim.SGD(
                weight_params,
                args.weight_opt_lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=args.nesterov,
            )
            return optimizer1, optimizer2
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
        )
    return optimizer, None


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"
    return log_base_dir.exists() or ckpt_base_dir.exists()

def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    args.rep_count = "/"
    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1
        args.rep_count = "/" + str(rep_count)
        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Epoch, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Setting\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{epoch}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{setting}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
