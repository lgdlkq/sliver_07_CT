#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :progress_bar.py
# @Date   :2019/4/4

import sys

import math


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode=None, epoch=None, total_epoch=None, current_loss=None,
                 current_acc=None, model_name=None,total=None, current=None,
                 width=30, symbol=">", output=sys.stderr):
        assert len(symbol) == 1

        self.mode = mode
        self.total = total
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_loss = current_loss
        self.current_acc = current_acc
        self.model_name = model_name

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss": self.current_loss,
            "current_acc": self.current_acc,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch
        }

        message = "\033[1;32;40m%(mode)s Epoch:%(epoch)3d/%(epochs)d %(bar)s\033[0m [Loss: %(current_loss)0.4f acc: %(current_acc)0.4f] %(current)3d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" % args
        self.write_message = "%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s  [loss %(current_loss)f acc: %(current_acc)f ]  %(current)d/%(total)d [ %(percent)3d%% ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self,time,loss,acc):
        h = math.floor(time/3600)
        m = math.floor((time-3600*h)/60)
        s = math.ceil(time-3600*h-60*m)
        self.current = self.total
        self()
        print(' time:%02d:%02d:%02d avg[loss:%0.4f acc:%0.4f]' % (h,m,s,loss,acc),file=self.output,end='')
        print("", file=self.output)
        # with open("../logs/%s.txt" % self.model_name, "a") as f:
        #     print(self.write_message, file=f)


if __name__ == "__main__":

    from time import sleep, time

    progress = ProgressBar("Train", total_epoch=100, model_name="test")

    for i in range(10):
        start = time()
        progress.total = 100
        progress.epoch = i
        progress.current_loss = 0.15
        progress.current_acc = 0.45
        for x in range(100):
            progress.current = x
            progress()
            sleep(0.05)
        progress.done(time()-start,0.15,0.2)
