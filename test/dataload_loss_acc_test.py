#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :dataload_loss_acc_test.py
# @Date   :2019/4/4

'''
乱，未整理结构，分解函数式组成
'''

from torch import optim
import time
from utils.utils import *
from dataset.data_laoders import *
from models.nets import SliverNet

class dense_block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(dense_block,self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel,out_channel,kernel_size=(3,3),padding=(1,1)),
            nn.Dropout(0.2)
        )
    def forward(self, *input):
        return self.layer(input[0])

class transitionDown(nn.Module):
    def __init__(self,in_channel):
        super(transitionDown,self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3),padding=1),
            nn.Dropout(0.2),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
    def forward(self, *input):
        return self.layer(input[0])

class transitionUp(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(transitionUp,self).__init__()
        self.layer = nn.ConvTranspose2d(in_channel, out_channel,stride=2,kernel_size=2)

    def forward(self, *input):
        x = input[0]
        return self.layer(x)

class DENSE_UNET(nn.Module):
    def __init__(self):
        super(DENSE_UNET,self).__init__()
        self.n_pool = 3
        self.growth_rate = 24
        self.layer_per_block = [3, 5, 7, 9, 7, 5, 3]
        self.nb_features = 48

        self.coven1 = nn.Conv2d(1,48,kernel_size=(7, 7), stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.db0_0 = dense_block(48,self.growth_rate)
        self.db0_1 = dense_block(72, self.growth_rate)
        self.db0_2 = dense_block(96, self.growth_rate)
        self.td0 = transitionDown(120)
        self.db1_0 = dense_block(120, self.growth_rate)
        self.db1_1 = dense_block(144, self.growth_rate)
        self.db1_2 = dense_block(168, self.growth_rate)
        self.db1_3 = dense_block(192, self.growth_rate)
        self.db1_4 = dense_block(216, self.growth_rate)
        self.td1 = transitionDown(240)
        self.db2_0 = dense_block(240, self.growth_rate)
        self.db2_1 = dense_block(264, self.growth_rate)
        self.db2_2 = dense_block(288, self.growth_rate)
        self.db2_3 = dense_block(312, self.growth_rate)
        self.db2_4 = dense_block(336, self.growth_rate)
        self.db2_5 = dense_block(360, self.growth_rate)
        self.db2_6 = dense_block(384, self.growth_rate)
        self.td2 = transitionDown(408)
        self.db3_0 = dense_block(408,self.growth_rate)
        self.db3_1 = dense_block(432, self.growth_rate)
        self.db3_2 = dense_block(456, self.growth_rate)
        self.db3_3 = dense_block(480, self.growth_rate)
        self.db3_4 = dense_block(504, self.growth_rate)
        self.db3_5 = dense_block(528, self.growth_rate)
        self.db3_6 = dense_block(552, self.growth_rate)
        self.db3_7 = dense_block(576, self.growth_rate)
        self.db3_8 = dense_block(600, self.growth_rate)
        self.tp0 = transitionUp(624,216)
        self.db4_0 = dense_block(624,self.growth_rate)
        self.db4_1 = dense_block(648, self.growth_rate)
        self.db4_2 = dense_block(672, self.growth_rate)
        self.db4_3 = dense_block(696, self.growth_rate)
        self.db4_4 = dense_block(720, self.growth_rate)
        self.db4_5 = dense_block(744, self.growth_rate)
        self.db4_6 = dense_block(768, self.growth_rate)
        self.tp1 = transitionUp(792,168)
        self.db5_0 = dense_block(408, self.growth_rate)
        self.db5_1 = dense_block(432, self.growth_rate)
        self.db5_2 = dense_block(456, self.growth_rate)
        self.db5_3 = dense_block(480, self.growth_rate)
        self.db5_4 = dense_block(504, self.growth_rate)
        self.tp2 = transitionUp(528, 120)
        self.db6_0 = dense_block(240,self.growth_rate)
        self.db6_1 = dense_block(264, self.growth_rate)
        self.db6_2 = dense_block(288, self.growth_rate)
        self.bn2 = nn.BatchNorm2d(312)
        self.relu2 = nn.ReLU(inplace=True)
        self.coven2 = nn.Conv2d(312,1,kernel_size=1)
        self.res = nn.Sigmoid()

    def forward(self, *input):
        x = input[0]
        skip_connections = []
        c0 = self.coven1(x)
        b0 = self.bn1(c0)
        r0 = self.relu1(b0)
        p0 = self.pool1(r0)
        db0_0 = self.db0_0(p0)
        db0 = torch.cat([p0,db0_0],dim=1)
        db0_1 = self.db0_1(db0)
        db0 = torch.cat([db0, db0_1], dim=1)
        db0_2 = self.db0_2(db0)
        db0 = torch.cat([db0, db0_2], dim=1)
        skip_connections.append(db0)
        td0 = self.td0(db0)
        db1_0 = self.db1_0(td0)
        db1 = torch.cat([td0,db1_0],dim=1)
        db1_1 = self.db1_1(db1)
        db1 = torch.cat([db1, db1_1], dim=1)
        db1_2 = self.db1_2(db1)
        db1 = torch.cat([db1, db1_2], dim=1)
        db1_3 = self.db1_3(db1)
        db1 = torch.cat([db1, db1_3], dim=1)
        db1_4 = self.db1_4(db1)
        db1 = torch.cat([db1, db1_4], dim=1)
        skip_connections.append(db1)
        td1 = self.td1(db1)
        db2_0 = self.db2_0(td1)
        db2 = torch.cat([td1, db2_0], dim=1)
        db2_1 = self.db2_1(db2)
        db2 = torch.cat([db2, db2_1], dim=1)
        db2_2 = self.db2_2(db2)
        db2 = torch.cat([db2, db2_2], dim=1)
        db2_3 = self.db2_3(db2)
        db2 = torch.cat([db2, db2_3], dim=1)
        db2_4 = self.db2_4(db2)
        db2 = torch.cat([db2, db2_4], dim=1)
        db2_5 = self.db2_5(db2)
        db2 = torch.cat([db2, db2_5], dim=1)
        db2_6 = self.db2_6(db2)
        db2 = torch.cat([db2, db2_6], dim=1)
        skip_connections.append(db2)
        td2= self.td2(db2)
        db3_0 = self.db3_0(td2)
        db3 = torch.cat([td2, db3_0], dim=1)
        db3_1 = self.db3_1(db3)
        db3 = torch.cat([db3, db3_1], dim=1)
        db3_2 = self.db3_2(db3)
        db3 = torch.cat([db3, db3_2], dim=1)
        db3_3 = self.db3_3(db3)
        db3 = torch.cat([db3, db3_3], dim=1)
        db3_4 = self.db3_4(db3)
        db3 = torch.cat([db3, db3_4], dim=1)
        db3_5 = self.db3_5(db3)
        db3 = torch.cat([db3, db3_5], dim=1)
        db3_6 = self.db3_6(db3)
        db3 = torch.cat([db3, db3_6], dim=1)
        db3_7 = self.db3_7(db3)
        db3 = torch.cat([db3, db3_7], dim=1)
        db3_8 = self.db3_8(db3)
        db3 = torch.cat([db3, db3_8], dim=1)
        skip_connections = list(reversed(skip_connections))
        tp0 = self.tp0(db3)
        tp0 = torch.cat([tp0, skip_connections[0]], dim=1)
        db4_0 = self.db4_0(tp0)
        db4 = torch.cat([tp0, db4_0], dim=1)
        db4_1 = self.db4_1(db4)
        db4 = torch.cat([db4, db4_1], dim=1)
        db4_2 = self.db4_2(db4)
        db4 = torch.cat([db4, db4_2], dim=1)
        db4_3 = self.db4_3(db4)
        db4 = torch.cat([db4, db4_3], dim=1)
        db4_4 = self.db4_4(db4)
        db4 = torch.cat([db4, db4_4], dim=1)
        db4_5 = self.db4_5(db4)
        db4 = torch.cat([db4, db4_5], dim=1)
        db4_6 = self.db4_6(db4)
        db4 = torch.cat([db4, db4_6], dim=1)
        tp1 = self.tp1(db4)
        tp1 = torch.cat([tp1, skip_connections[1]], dim=1)
        db5_0 = self.db5_0(tp1)
        db5 = torch.cat([tp1,db5_0],dim=1)
        db5_1 = self.db5_1(db5)
        db5 = torch.cat([db5, db5_1], dim=1)
        db5_2 = self.db5_2(db5)
        db5 = torch.cat([db5, db5_2], dim=1)
        db5_3 = self.db5_3(db5)
        db5 = torch.cat([db5, db5_3], dim=1)
        db5_4 = self.db5_4(db5)
        db5 = torch.cat([db5, db5_4], dim=1)
        tp2 = self.tp2(db5)
        tp2 = torch.cat([tp2, skip_connections[2]], dim=1)
        db6_0 = self.db6_0(tp2)
        db6 = torch.cat([tp2,db6_0],dim=1)
        db6_1 = self.db6_1(db6)
        db6 = torch.cat([db6, db6_1], dim=1)
        db6_2 = self.db6_2(db6)
        db6 = torch.cat([db6, db6_2], dim=1)
        b1 = self.bn2(db6)
        r1 = self.relu2(b1)
        c1 = self.coven2(r1)
        res = self.res(c1)
        return res

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Operation():
    def __init__(self,model=SliverNet(),criterion = DiceLoss()):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optim.Adam(model.parameters())

    def train(self,num_epochs=20):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dt_size = len(get_train_data().dataset)
            epoch_loss = 0
            step = 0
            count = 0
            start = time.time()
            for i,(x, y) in enumerate(get_train_data()):
                step += 1
                count += 1
                inputs = x.to(device)
                labels = y.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(labels, outputs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # writer.add_scalar('batch_train_loss', loss.item(), count)
                print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // get_train_data().batch_size + 1, loss.item()))
            # writer.add_scalar('epoch_train_loss',epoch_loss,epoch)
            print("epoch %d loss:%0.3f  time cost:%fs" % (epoch, epoch_loss, time.time() - start))
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), 'd_unet_weights_%d.pth' % epoch)

    def test(self,ckp):
        self.model.load_state_dict(torch.load(ckp, map_location='cuda:0'))
        self.model.eval()
        acc = Accessory()
        i = 0
        all_loss = 0
        with torch.no_grad():
            for x, labels in get_test_data():
                i += 1
                print(i, end='  ')
                x = x.to(device)
                y = self.model(x)
                labels = labels.to(device)
                accessory = acc.iou(labels,y)
                loss = self.criterion(labels,y)
                all_loss += loss.item()
                # l = torch.squeeze(labels).cpu().numpy()
                # l = cv2.resize(l, (512, 512))
                # cv2.imshow('l',l)
                # img_y = torch.squeeze(y).cpu().numpy()
                # img_y = cv2.resize(img_y,(512,512))
                # cv2.imshow('y',img_y)
                # cv2.waitKey(0)
                print("loss:%0.4f  acc:%0.4f" % (loss.item(),accessory))
            print("all acc:%0.3f" % ((1 - all_loss / i) * 100) + '%')

if __name__ == '__main__':
    ckp = 'G:/PythonTrainFaile/u_net_liver/d_unet_weights_19.pth'
    op = Operation()
    op.train()
    # op.test(ckp)



