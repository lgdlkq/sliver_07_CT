import sys
from PyQt5.QtGui import QCursor

from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from qtpy import QtCore, QtGui


class Wait(QWidget):
    def __init__(self):
        super(Wait,self).__init__()
        self.label = QLabel(self)
        self.label.setWindowFlags(QtCore.Qt.FramelessWindowHint)#无边框
        self.label.setAttribute(QtCore.Qt.WA_TintedBackground)#背景透明
        self.movie = QtGui.QMovie('../resource/wait.gif')
        self.movie.setCacheMode(QtGui.QMovie.CacheAll)#无限循环
        self.movie.setSpeed(100)
        self.label.setMovie(self.movie)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(QtCore.Qt.ArrowCursor))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    wait=Wait()
    wait.movie.start()
    # wait.label.show()
    wait.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    wait.show()
    sys.exit(app.exec_())