import torch
import tqdm
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import constrainScoreByWhole
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
__all__ = ["train", "validate", "modifier"]

def calculateGrad(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1/(args.K-1)*(fn_list[0] - fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(fn_list[1] - fn_avg)*getattr(m, 'stored_mask_1')

def train(train_loader, model, criterion, optimizer, epoch, args, writer, weight_opt=None):
    losses = AverageMeter("Loss", ":.3f")
    original_losses = AverageMeter("Orig Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    v_meter = AverageMeter("v", ":6.4f")
    max_score_meter = AverageMeter("max_score", ":6.4f")
    l = [losses, original_losses, top1, top5, v_meter, max_score_meter]
    progress = ProgressMeter(
        len(train_loader),
        l,
        prefix=f"Epoch: [{epoch}]",
    )
    model.train()
    args.discrete = False
    args.val_loop = False
    args.num_batches = len(train_loader)
    for i, (image, target) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        l, ol, gl, al, a1, a5, ll = 0, 0, 0, 0, 0, 0, 0
        if optimizer is not None:
            optimizer.zero_grad()
        if weight_opt is not None:
            weight_opt.zero_grad()
        fn_list = []
        for j in range(args.K):
            args.j = j
            output = model(image)
            original_loss = criterion(output, target)
            loss = original_loss/args.K
            fn_list.append(loss.item()*args.K)
            loss.backward(retain_graph=True)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            l = l + loss.item()
            ol = ol + original_loss.item() / args.K
            a1 = a1 + acc1.item() / args.K
            a5 = a5 + acc5.item() / args.K
        fn_avg = l
        if not args.finetuning:
            if args.conv_type == "VRPGE":
                calculateGrad(model, fn_avg, fn_list, args)
        losses.update(l, image.size(0))
        original_losses.update(ol, image.size(0))
        top1.update(a1, image.size(0))
        top5.update(a5, image.size(0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        if optimizer is not None:
            optimizer.step()
        if weight_opt is not None:
            weight_opt.step()
        if args.conv_type == "VRPGE":
            if not args.finetuning:
                with torch.no_grad():
                    constrainScoreByWhole(model, v_meter, max_score_meter)
        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(len(train_loader))
    progress.write_to_tensorboard(writer, prefix="train", global_step=epoch)
    return top1.avg, top5.avg

def validate(val_loader, model, criterion, args, writer, epoch):
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(len(val_loader), [losses, top1, top5], prefix="Test: ")
    args.val_loop = True
    if args.use_running_stats:
        model.eval()
    if writer is not None:
        for n, m in model.named_modules():
            if hasattr(m, "scores") and m.prune:
                writer.add_histogram(n, m.scores)
    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader)):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            args.discrete = False
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            if i % args.print_freq == 0:
                progress.display(i)
        progress.display(len(val_loader))
        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
    return top1.avg, top5.avg, losses.avg


def modifier(args, epoch, model):
    return
