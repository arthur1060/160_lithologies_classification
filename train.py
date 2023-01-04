import os
import math
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn

from model.model41 import MetaMo
from my_dataset import MyDataSet, SHDataSets
from utils import evaluate, read_split_datas

from tqdm import tqdm


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label = read_split_datas(".\\data\\train")
    test_images_path, test_images_label = read_split_datas(".\\data\\test")

    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_dataset = SHDataSets(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    batch_size = args.batch_size
    nw = 4
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)

    model = MetaMo(num_classes=args.num_classes).to(device)
    model = nn.DataParallel(model)
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict, strict=False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1 * args.lr, args.lr, step_size_up=1500,
                                                  step_size_down=2000, mode='triangular', gamma=1.0,
                                                  scale_fn=None, scale_mode='cycle', cycle_momentum=False,
                                                  base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
    loss_function = nn.CrossEntropyLoss()
    train_num = len(train_loader.dataset)
    for epoch in range(args.epochs):
        # train
        right_num = torch.zeros(1).to(device)
        mean_loss = torch.zeros(1).to(device)
        model.train()
        data_loader = tqdm(train_loader)
        for step, data in enumerate(data_loader):
            images, labels = data
            optimizer.zero_grad()
            pred = model(images.to(device))

            loss = loss_function(pred, labels.to(device))
            loss.backward()
            pred1 = torch.max(pred, dim=1)[1]
            right_num += torch.eq(pred1, labels.to(device)).sum()
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

            train_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 4))

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            scheduler.step()

        acc_train = round(right_num.item() / train_num, 4)
        #print("total_num",total_num)

        acc_test = evaluate(model=model,
                            data_loader=test_loader,
                            device=device)
        print("Epoch:{}, train accuracy:{},test accuracy:{}".format(epoch,acc_train,acc_test))

        tags = ["loss", "accuracy_test","accuracy_val", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc_train, epoch)
        tb_writer.add_scalar(tags[2], acc_test, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/Meta1-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=160)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weights', type=str, default= ".\\trained_model\\Meta1-61.pth",
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
