# train_diffusion.py

import torch.nn as nn
import torch
import utils.utils
import wandb  

def compute_loss(mse_criterion, x_t_from_diffusion, x_t_from_vit):
    """
    Compute the total loss as the sum of MSE losses between Diffusion and ViT outputs.
    
    Parameters:
        mse_criterion (nn.Module): MSE loss function.
        diffusion_layer_outputs (list of tensors): Sampled outputs from Diffusion model layers.
        vit_layer_outputs (list of tensors): Outputs from ViT model layers.
    
    Returns:
        total_loss (Tensor): Sum of MSE losses across all layers.
        layer_losses (dict): Dictionary of individual layer MSE losses.
    """
    total_mse = 0

    for layer_idx, (diff_out, vit_out) in enumerate(zip(x_t_from_diffusion, x_t_from_vit)):
        # Compute MSE loss between Diffusion output and ViT output
        mse_loss = mse_criterion(diff_out, vit_out)
        total_mse += mse_loss
    
    return total_mse


def train(train_loader, diffusion_model, optimizer, epoch, logger, args, vit_model):
    """
    Train the Diffusion model by aligning its layers with the ViT model's layers using MSE loss.
    
    Parameters:
        train_loader (DataLoader): Training data loader.
        diffusion_model (nn.Module): Diffusion model to be trained.
        optimizer (Optimizer): Optimizer for the Diffusion model.
        epoch (int): Current epoch number.
        logger (Logger): Logger for logging information.
        args (Namespace): Command-line arguments.
        vit_model (nn.Module): Pre-trained ViT model for layer alignment.
    """
    diffusion_model.train()
    vit_model.eval()  # Ensure ViT is in evaluation mode

    # Freeze ViT model parameters
    # for param in diffusion_model.parameters():
    #     param.requires_grad = False

    for param in vit_model.parameters():
        param.requires_grad = False

    # for param in vit_model.fc.parameters():
    #     param.requires_grad = True

    # Define loss function
    mse_criterion = nn.MSELoss() #to be uncomment

    # ce_criterion = nn.CrossEntropyLoss()

    # Initialize training logs
    train_log = {
        'Tot. Loss': utils.utils.AverageMeter(),
        'LR': utils.utils.AverageMeter(),
    }

    msg = '####### --- Training Epoch {:d} --- #######'.format(epoch)
    logger.info(msg)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = vit_model._to_words(inputs)
        output = vit_model.emb(output)
        output = output + vit_model.pos_emb
        output = diffusion_model(output)
        output = vit_model.fc(output.mean(1))

        # loss = ce_criterion(output, targets)
        with torch.no_grad(): #to be uncomment
            out, score_list, Lambda_inv_list, kl_list, x_t_from_ViT = vit_model(inputs)
        x_t_from_diffusion = diffusion_model(x_t_from_ViT, train=True)
        loss = compute_loss(mse_criterion, x_t_from_diffusion, x_t_from_ViT) #to be uncomment
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

        train_log['Tot. Loss'].update(loss.item(), inputs.size(0))
        train_log['LR'].update(lr, inputs.size(0))

        if i % 100 == 99:
            log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [
                key + ': {:.2f}'.format(train_log[key].avg) for key in train_log if key != 'LR'
            ]
            msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
            logger.info(msg)
            for key in train_log:
                train_log[key] = utils.utils.AverageMeter()

    # Replace writer.add_scalar with wandb.log
    wandb.log({f"Train/{key}": train_log[key].avg for key in train_log}, step=epoch)
    