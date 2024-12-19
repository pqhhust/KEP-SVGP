import model.vit_cifar
import model.diffusion


def get_model(model_name, nb_cls, logger, args):
  
    if model_name == "vit_cifar":
        net = model.diffusion.vit_cifar(args=args, attn_type=args.attn_type, num_classes=nb_cls, ksvd_layers=args.ksvd_layers, low_rank=args.low_rank, rank_multi=args.rank_multi).cuda()
    if model_name == "diffusion":
        net = model.diffusion.Diffusion()
    msg = 'Using {} ...'.format(model_name)
    logger.info(msg)
    return net