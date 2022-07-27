from tqdm import tqdm
import numpy as np
import torch
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.utils.parser_util import get_parser
from core.engines import train, test, eval
from utils.init_engine import init_seed, init_dataloader, init_protonet, init_optim, init_lr_scheduler


def main():
    # Get parameters
    options = get_parser().parse_args()
    
    # Init output path of model
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    # Init data
    train_loader = init_dataloader(options, 'train')
    val_loader = init_dataloader(options, 'val')
    # test_loader = init_dataloader(options, 'test')

    # init model and engine
    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    
    res = train(opt=options,
                tr_dataloader=train_loader,
                val_dataloader=val_loader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)

    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_loader,
         model=model)

    model.load_state_dict(best_state)
    
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_loader,
         model=model)


if __name__ == '__main__':
    main()