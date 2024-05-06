import torch
import numpy as np
import os
from collections import defaultdict


def adjust_lr(optimizer, epoch, sched_type, lr, lr_decay, steps):
    if sched_type == 'step':
        steps = [int(k) for k in steps.split(";")]
        lr = lr * lr_decay ** len([x for x in steps if x < epoch])
    elif sched_type == 'exp':
        lr = lr * lr_decay**epoch

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optim(params, lr, momentum=0.9):
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    return optimizer


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False


def thaw(m, mode='all'):
    freeze(m)
    for k, p in m.named_parameters():
        if mode == 'all':
            p.requires_grad = True
        elif mode == 'bias':
            if 'bias' in k:
                p.requires_grad = True
        elif mode == 'last':
            if 'blocks.11' in k or k == 'norm.weight' or k == 'norm.bias':
                p.requires_grad = True
        elif mode == 'last2':
            if 'blocks.11' in k or k == 'norm.weight' or k == 'norm.bias' or 'blocks.10' in k:
                p.requires_grad = True
        else:
            raise Exception("No such thaw mode")


def collect_params(m, quiet=False):
    parameters = []
    for k, p in m.named_parameters():
        if p.requires_grad:
            if not quiet:
                print(k, p.data.shape)
            parameters.append(p)
    return parameters


def get_loss(args):
    if args.loss == 'mse':
        return torch.nn.MSELoss()
    elif args.loss == 'mae':
        return torch.nn.L1Loss()
    else:
        raise ValueError()


def l2_penalty(model):
    l2_loss = 0
    for k, p in model.named_parameters():
        if p.requires_grad:
            l2_loss += p.pow(2).sum()
    return l2_loss ** 0.5



def distill(args, model, dataset):
    model.train()
    thaw(model, mode=args.unfreeze)
    params = collect_params(model)
    optimizer = get_optim(params, args.lr)
    # Use MSE for now
    loss_fn = get_loss(args)
    reg_fn = l2_penalty

    for epoch in range(args.epochs):
        adjust_lr(optimizer, epoch, args.sched_type, args.lr, args.lr_decay, args.steps)
        epoch_stats = defaultdict(list)
        for batch_idx, (frame, features) in enumerate(dataset):
            frame, features = frame.cuda(), features.cuda()
            out = model.get_intermediate_layers(frame.unsqueeze(0), n=1)[0]
            out = out[:, 1:, :]  
            reg = reg_fn(model)
            emp_risk = loss_fn(out, features)
            loss = emp_risk + args.wd * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_stats['loss'].append(loss.item())
            epoch_stats['emp_risk'].append(emp_risk.item())
            epoch_stats['reg'].append(reg.item())
            print(f"[Batch {batch_idx}/{len(dataset)}]: Loss={loss:.5f}  Emp_risk={emp_risk:.5f}  Reg={reg:.5f}", end='\r', flush=True)

        print()

        print_str = f"Epoch {epoch}/{args.epochs}:"
        for k,v in epoch_stats.items():
            print_str += f" {k}:{np.mean(v):.4f} "
        print(print_str)

    if args.model_save_dir != '':
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        model_save_path =  os.path.join(args.model_save_dir, 'model-distill.pt')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    return model


