import argparse
import os.path as osp
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data.mini_imagenet import MiniImageNet
from core.data.batch_sampler import CategoriesSampler
from core.model.convnet_4 import Convnet
from core.utils.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, LoadConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='../config/my_cfg.json')
    args = parser.parse_args()
    config = LoadConfig(args.config_path)

    set_gpu(config.GPU)
    ensure_path(config.SAVE_PATH)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      config.TRAIN_WAY, config.SHOT + config.QUERY)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=4, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    config.TEST_WAY, config.SHOT + config.QUERY)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=4, pin_memory=True)

    model = Convnet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(config.SAVE_PATH, name + '.pth'))
    
    trlog = {}
    trlog['config'] = vars(config)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, config.EPOCHS + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = config.SHOT * config.TRAIN_WAY
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.SHOT, config.TRAIN_WAY, -1).mean(dim=0)

            label = torch.arange(config.TRAIN_WAY).repeat(config.QUERY)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = config.SHOT * config.TEST_WAY
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.SHOT, config.TEST_WAY, -1).mean(dim=0)

            label = torch.arange(config.TEST_WAY).repeat(config.QUERY)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(config.SAVE_PATH, 'trlog'))

        save_model('epoch-last')

        if epoch % config.SAVE_EACH_EPOCH == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / config.EPOCHS)))

