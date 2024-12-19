import argparse

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Kernel-Eigen Pair Sparse Variational Gaussian Processes',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nb-epochs', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--nb-worker', default=4, type=int, help='Nb of workers')
    parser.add_argument('--seed', default=0, type=int, help='Set seed for reproducibility')
    parser.add_argument('--wandb-key', default=None, type=str, help='Set wandb key for tracking progresses')
    
    ## Model
    parser.add_argument('--model', default='vit_cifar', type=str, choices = ['vit_cifar', 'diffusion'], help='Models name to use')
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--hdim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=12)

    # KEP-SVGP-attention
    parser.add_argument('--ksvd-layers', type=int, default=1, help='Number of ksvd layers applied to the transformer')
    parser.add_argument('--attn-type', default='kep_svgp', type=str, choices = ['kep_svgp', 'softmax'], help='Type of attention')
    parser.add_argument('--concate', action='store_true', help='whether to use [e(x),r(x)] instead of (e(x)+r(x))')  
    parser.add_argument('--eta-ksvd', type=float, default=0.1, help='coefficient of the KSVD regularization')
    parser.add_argument('--eta-kl', type=float, default=1.0, help='coefficient of the KL divergence regularization')
    parser.add_argument('--low_rank', type=int, default=10, help='Number of dimension the low rank method projected to')
    parser.add_argument('--rank_multi', type=int, default=10, help='low rank dimension * rank_multi')

    ## optimizer 
    parser.add_argument('--lr', default=1e-3, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup-epoch", default=5, type=int)

    ## nb of run 
    parser.add_argument('--nb-run', default=1, type=int, help='Run n times, in order to compute std')
    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    ## dataset setting
    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")
    Cifar10 = subparsers.add_parser("Cifar10",
                                    description='Dataset parser for training on Cifar10',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on Cifar10")
    Cifar10.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    Cifar10.add_argument("--train-dir", type=str, default='./data/CIFAR10/train', help="Cifar10 train directory")
    Cifar10.add_argument("--val-dir", type=str, default='./data/CIFAR10/val', help="Cifar10 val directory")
    Cifar10.add_argument("--test-dir", type=str, default='./data/CIFAR10/test', help="Cifar10 test directory")
    Cifar10.add_argument("--corruption-dir", type=str, default='./data', help="Cifar10-C directory")
    Cifar10.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10")

    Cifar100 = subparsers.add_parser("Cifar100",
                                     description='Dataset parser for training on Cifar100',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     help="Dataset parser for training on Cifar100")
    Cifar100.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
    Cifar100.add_argument("--train-dir", type=str, default='./data/CIFAR100/train', help="Cifar100 train directory")
    Cifar100.add_argument("--val-dir", type=str, default='./data/CIFAR100/val', help="Cifar100 val directory")
    Cifar100.add_argument("--test-dir", type=str, default='./data/CIFAR100/test', help="Cifar100 test directory")
    Cifar100.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100")

    return parser.parse_args()
