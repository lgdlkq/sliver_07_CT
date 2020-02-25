import sys

import os
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.resize(600, 570)
        self.setWindowTitle("label显示图片")
        self.label = QLabel(self)
        self.label.setFixedSize(512, 512)
        self.label.move(10, 30)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        # pic = QtGui.QPixmap(os.path.dirname(os.getcwd())+"/resource/13.png").scaled(self.label.width(), self.label.height())
        # self.label.setPixmap(pic)
        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(10, 30)
        btn.clicked.connect(self.openimage)
    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(imgName[-3:],'  ',imgType,'   ',type(imgType))
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())