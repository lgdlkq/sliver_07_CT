#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :operation.py
# @author :雷国栋
# @Date   :2019/4/8
import numpy as np
import time
import cv2
import torch
from tensorboardX import SummaryWriter

from dataset.data_ready import ReadyData
from dataset.data_laoders import *
from models.nets import SliverNet
from utils import utils as u
from utils.progress_bar import *
from utils.AdaBound import *

random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
torch.cuda.manual_seed_all(configs.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpus
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold = 2
if not os.path.exists(configs.weights):
    os.mkdir(configs.weights)
if not os.path.exists(configs.best_models):
    os.mkdir(configs.best_models)
if not os.path.exists(configs.logs):
    os.mkdir(configs.logs)
if not os.path.exists(
        configs.weights + configs.model_name + os.sep + str(fold) + os.sep):
    os.makedirs(
        configs.weights + configs.model_name + os.sep + str(fold) + os.sep)
if not os.path.exists(
        configs.best_models + configs.model_name + os.sep + str(fold) + os.sep):
    os.makedirs(
        configs.best_models + configs.model_name + os.sep + str(fold) + os.sep)


class Operater():
    def __init__(self, model=SliverNet(), criterion=u.DiceLoss(), opt=None):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        if opt is None:
            # self.opt = AdaBound(model.parameters(), lr=configs.lr,
            #                     final_lr=configs.final_lr,
            #                     weight_decay=configs.weight_decay)
            # self.opt = torch.optim.RMSprop(self.model.parameters(),lr=1e-5,momentum=0.9,weight_decay=1e-4)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=configs.lr,
                                        weight_decay=configs.weight_decay)
        else:
            self.opt = opt.to(device)

        self.acc = u.Accessory()

    def train(self,resume = False):
        global fold
        start_epoch = 0
        best_precision = 0
        if resume:
            checkpoint = torch.load(
                configs.best_models + configs.model_name + os.sep + str(
                    fold) + '/model_best.pth.tar')
            start_epoch = checkpoint["epoch"]
            fold = checkpoint["fold"]
            best_precision = checkpoint["best_precision"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.opt.load_state_dict(checkpoint["optimizer"])

        total = len(get_train_data())
        for epoch in range(start_epoch, configs.epochs):
            train_progressor = ProgressBar(mode="Train", epoch=epoch,
                                           total_epoch=configs.epochs,
                                           model_name=configs.model_name,
                                           total=total)
            epoch_loss = 0
            start = time.time()
            epoch_acc = 0
            self.model.train()
            if epoch < 20:
                td = get_train_data()
            else:
                td = get_train_data(True)
            for i, (x, y) in enumerate(td):
                x = x.to(device)
                y = y.to(device)
                train_progressor.current = i
                self.opt.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(y, outputs)
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()
                acc = self.acc.iou(y, outputs)
                epoch_acc += acc
                time_cost = time.time() - start
                train_progressor.current_loss = loss.item()
                train_progressor.current_acc = acc
                train_progressor()
            iter = total
            train_progressor.done(time_cost, epoch_loss / iter,
                                  epoch_acc / iter)
            # writer.add_scalar('avg_epoch_train_loss', epoch_loss / iter, epoch)
            # writer.add_scalar('avg_epoch_train_acc', epoch_acc / iter, epoch)
            val_loss, val_acc = self.evaluate(epoch)
            # writer.add_scalar('avg_epoch_val_loss', val_loss, epoch)
            # writer.add_scalar('avg_epoch_val_acc', val_acc, epoch)
            is_best = val_acc > best_precision
            best_precision = max(val_acc, best_precision)
            u.save_checkpoint({
                "epoch": epoch + 1,
                "model_name": configs.model_name,
                "state_dict": self.model.state_dict(),
                "best_precision": best_precision,
                "optimizer": self.opt.state_dict(),
                "fold": fold,
                "valid_loss": val_loss,
                "valid_acc": val_acc,
            }, is_best, fold)

    def evaluate(self, epoch):
        val_loss, val_acc = 0, 0
        total = len(get_valid_data())
        val_progressor = ProgressBar(mode="Valid", epoch=epoch,
                                     total_epoch=configs.epochs,
                                     model_name=configs.model_name,
                                     total=total)
        self.model.eval()
        if epoch < 20:
            vd = get_valid_data()
        else:
            vd = get_valid_data(True)
        with torch.no_grad():
            start = time.time()
            for i, (x, y) in enumerate(vd):
                x = x.to(device)
                y = y.to(device)
                outputs = self.model(x)
                val_progressor.current = i
                loss = self.criterion(y, outputs)
                acc = self.acc.iou(y, outputs)
                val_loss += loss.item()
                val_acc += acc
                time_cost = time.time() - start
                val_progressor.current_loss = loss.item()
                val_progressor.current_acc = acc
                val_progressor()
            iter = total
            val_progressor.done(time_cost, val_loss / iter, val_acc / iter)
        return val_loss / iter, val_acc / iter

    def test(self, td, o):
        test_loss, test_acc = 0, 0
        total = len(td)
        test_progressor = ProgressBar(mode="Test", epoch=0, total_epoch=1,
                                      model_name=configs.model_name,
                                      total=total)
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            for i, (x, y) in enumerate(td):
                x = x.to(device)
                y = y.to(device)
                outputs = self.model(x)
                # import cv2
                # l = torch.squeeze(y).cpu().numpy()
                # l = cv2.resize(l, (512, 512))
                # cv2.imshow('l',l)
                # pre_y = torch.squeeze(outputs).cpu().numpy()
                # pre_y = cv2.resize(pre_y,(512,512))
                # cv2.imshow('y',pre_y)
                # cv2.waitKey(0)
                test_progressor.current = i
                loss = self.criterion(y, outputs)
                acc = self.acc.ious(y, outputs).cpu().numpy()
                test_loss += loss.item()
                test_acc += acc
                # print(i,'  ',acc,'  ',test_acc)
                time_cost = time.time() - start
                test_progressor.current_loss = loss.item()
                test_progressor.current_acc = acc
                test_progressor()
                # writer.add_scalar(o+'every_test_loss', loss.item(), i)
                # writer.add_scalar(o+'every_test_acc', acc, i)
            iter = total

            test_progressor.done(time_cost, test_loss / iter, test_acc / iter)
            print(test_acc)
            print(test_acc / iter)
        # return test_loss / iter, test_acc / iter

    def load_model(self):
        best_model = torch.load(
            configs.best_models + configs.model_name + os.sep + str(
                fold) + '/model_best.pth.tar')
        self.model.load_state_dict(best_model["state_dict"])
        self.model.eval()

    def predict_pic(self,x):
        with torch.no_grad():
            x = x.to(device)
            outputs = self.model(x)
            outputs = torch.round(outputs)
            pre_y = torch.squeeze(outputs).cpu().numpy()
            pre_y = cv2.resize(pre_y,(512,512))
            # cv2.imshow('y',pre_y)
            # cv2.waitKey(0)
            return pre_y

if __name__ == '__main__':
    model = SliverNet()
    # # writer = SummaryWriter(log_dir=configs.tf_logs, comment='SliverNet')
    # # writer.add_graph(model,(torch.rand((1,1,512,512)),))
    op = Operater(model=model)
    # op.train()
    op.load_model()
    # # torch.save(op.model.state_dict(), 'weights.pth')
    # # op.test(get_test_data(True), 'tr')
    op.test(get_test_data(), 'o')
    # # writer.close()

    # import cv2
    # path = r'F:\sliver\vol\17315.png'
    # path1 = r'F:\sliver\vol\14.png'
    # pic1 = cv2.imread(path1,cv2.IMREAD_UNCHANGED)
    # pic = cv2.imread(path,cv2.IMREAD_UNCHANGED)

    # import SimpleITK as sitk
    # import math
    #
    # path1 = r'F:\sliver_07\segmentation-2.nii'
    # ds1 = sitk.ReadImage(path1)
    # image1 = sitk.GetArrayFromImage(ds1)
    # image1[image1 == 2] = 1
    # image1 = image1.astype(dtype=np.float32) * 255.
    #
    # path = r'F:\sliver_07\volume-2.nii'  # segmentation-0.nii
    # ds = sitk.ReadImage(path)
    # image = sitk.GetArrayFromImage(ds)
    # image = image.astype(np.uint8)
    # op = Operater()
    # start = time.time()
    # print(start)
    # op.load_model()
    # print('load model cost: %f' % (time.time()-start))
    # r_dset = ReadyData()
    # start = time.time()
    # for i in range(math.ceil(image.shape[0]*0.8),image.shape[0]):
    #     print(i)
    #     cv2.imshow('z',image1[i])
    #     r_dset.update_data(image[i])
    #     op.predict_pic(r_dset.getitem().float())
    # # r_dset.update_data(pic)
    # # op.predict_pic(r_dset.getitem())
    # # r_dset.update_data(pic1)
    # # op.predict_pic(r_dset.getitem())
    # print('predict cost: %f' % (time.time() - start))
