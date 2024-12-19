import torch
import torch.nn as nn
import torch.backends.cudnn
import wandb

import os
import json 

import train_diffusion
import valid_diffusion

import model.get_model
import data.dataset
import utils.option

import warmup_scheduler

args = utils.option.get_args_parser()

if args.attn_type == 'softmax':
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}")
elif args.attn_type == 'kep_svgp':
    save_path = os.path.join(
        args.save_dir,
        f"{args.dataset}_{args.attn_type}_vit_cifar_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}"
    )

if not os.path.exists(save_path):
    os.makedirs(save_path)

wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')
wandb.init(project='Diffusion-KEP-SVGP', config=vars(args))

# Set seed everything
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

logger = utils.utils.get_logger(save_path)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

train_loader, val_loader, _, nb_cls = data.dataset.get_loader(
    args.dataset, args.train_dir, args.val_dir, args.test_dir, args.batch_size
)

for run in range(args.nb_run):
    prefix = f'{run + 1} / {args.nb_run} Running'
    logger.info(100*'#' + '\n' + prefix)

    ## define model
    net = model.get_model.get_model(args.model, nb_cls, logger, args)
    # net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{run + 1}_diffusion.pth')))
    print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    net.cuda()
    pretrained_ViT = model.get_model.get_model('vit_cifar', nb_cls, logger, args)
    pretrained_ViT.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{run + 1}.pth')))
    pretrained_ViT.cuda()
    
    ## define optimizer with warm-up
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.nb_epochs, eta_min=args.min_lr
    )
    scheduler = warmup_scheduler.GradualWarmupScheduler(
        optimizer,
        multiplier=1.,
        total_epoch=args.warmup_epoch,
        after_scheduler=base_scheduler
    )
    
    ## make logger
    best_acc, best_auroc, best_aurc = 0, 0, 1e6

    ## start training
    for epoch in range(args.nb_epochs):
        train_diffusion.train(train_loader, net, optimizer, epoch, logger, args, pretrained_ViT)
        
        scheduler.step()
        # optimizer.step()

        # validation
        net_val = net
        res = valid_diffusion.validation(val_loader, net_val, args, pretrained_ViT) 
        log = [f"{key}: {res[key]:.3f}" for key in res]
        msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
        logger.info(msg)

        wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

        if res['Acc.'] > best_acc:
            acc = res['Acc.']
            msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
            logger.info(msg)
            best_acc = acc
            torch.save(net_val.state_dict(), os.path.join(save_path, f'best_acc_net_{run+1}_diffusion_mlp_1.pth'))
            # torch.save(pretrained_ViT.state_dict(), os.path.join(save_path, f'best_acc_net_{run + 1}_vit_fc.pth'))
        
        if res['AUROC'] > best_auroc:
            auroc = res['AUROC']
            msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
            logger.info(msg)
            best_auroc = auroc
            torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}_diffusion.pth'))
    
        if res['AURC'] < best_aurc:
            aurc = res['AURC']
            msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
            logger.info(msg)
            best_aurc = aurc
            torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}_diffusion.pth'))

wandb.finish()