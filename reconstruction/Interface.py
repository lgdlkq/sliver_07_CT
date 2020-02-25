import os
import time
from PyQt5 import QtWidgets

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import *

os.environ['ETS_TOOLKIT'] = 'qt4'
# 默认情况下，将使用Pyside绑定。如果希望使用pyqt绑定，则需要将qt_api环境变量设置为“pyqt”
# os.environ['QT_API'] = 'pyqt'

# 为了能够使用pyside或pyqt4，并且不会与特性冲突。我们需要从pyface.qt导入qtgui和qtcore
from pyface.qt import QtGui, QtCore
# 或者，可以绕过这一行，但需要确保在导入pyqt之前执行以下行
#   import sip
#   sip.setapi('QString', 2)

from traits.api import HasTraits, Instance, on_trait_change,Str,Range
from traitsui.api import View, Item,Group
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
import SimpleITK as sitk
import PIL.Image as Image
import numpy as np
from operation import Operater
from dataset.data_ready import ReadyData
import math

################################################################################
class Visualization(HasTraits):
    path = Str
    data = None
    op = Operater()
    start = time.time()
    op.load_model()
    print('load model cost: %f' % (time.time() - start))
    r_dset = ReadyData()
    x = None
    y = None
    z = None
    opacity = Range(0, 100, 100)
    # the layout of the dialog screated 创建的对话框布局
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=512, width=512, show_label=False),
                Group('_','opacity'),
                resizable=True  # 需要用父窗口小部件调整大小
                )
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        # 打开视图时调用此函数。当视图尚未打开时，不会填充场景，因为某些VTK功能需要GLContext。
        # 可以在嵌入的场景上进行正常的MLAB调用。
        print('*_*'*10)
        if self.data is None:
            return
        print('-*-'*10)
        self.plot = self.scene.mlab.contour3d(self.x,self.y,self.z,self.data, color=(0.53, 0, 0.098), transparent=True,
                                  opacity=1.0)

    @on_trait_change('opacity')
    def update_opacity(self):
        s = self.scene.mlab.gcf()
        source = s.children[0]
        manager = source.children[0]
        surface = manager.children[0]
        surface.actor.property.opacity = self.opacity*1.0/100

    def update_data(self, path):
        self.path = path
        self.data = []
        if path == '' or path == None:
            return
        elif path[-4:] == '.nii':
            ds = sitk.ReadImage(path)
            img = sitk.GetArrayFromImage(ds)
            img = img.astype(np.uint8)
            img = img[46:]
            lz = img.shape[0]
            lx = img.shape[1]
            ly = img.shape[2]
            spacing = ds.GetSpacing()  # x, y, z
            self.x, self.y, self.z = np.mgrid[
                      -lz * spacing[2] / 2:lz * spacing[2] / 2:spacing[2],
                      -lx * spacing[0] / 2:lx * spacing[0] / 2:spacing[0],
                      -ly * spacing[1] / 2:ly * spacing[1] / 2:spacing[1]]
            start = time.time()
            for i in range(img.shape[0]):
                self.r_dset.update_data(img[i])
                self.data.append(self.op.predict_pic(self.r_dset.getitem().float()))
            self.data = np.array(self.data)
            print(lz)
            print(self.x.shape)
            print(self.y.shape)
            print(self.z.shape)
            print(self.data.shape)
            print('predict cost: %f' % (time.time() - start))
        else:
            self.data = []
            filelist = os.listdir(path)
            types = '.' + filelist[0].split('.')[1]
            for i in range(len(filelist)):
                image = Image.open(path + '/' + str(i) + types)
                image = np.array(image)
                self.data.append(image)
            self.data = np.array(self.data)
            self.x, self.y, self.z = np.mgrid[
                                     -len(filelist) * 1 / 2:len(filelist) * 1 / 2:
                                     1,
                                     -512 * 0.72 / 2:512 * 0.72 / 2:
                                     0.72,
                                     -512 * 0.72 / 2:512 * 0.72 / 2:
                                     0.72]

################################################################################
# 包含可视化的qwidget，纯pyqt4代码。
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        lay = QtGui.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.visualization = Visualization()
        # 编辑特性调用将生成要嵌入的小部件。
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        lay.addWidget(self.ui)
        self.ui.setParent(self)

    def update_data(self, path):
        self.visualization.update_data(path)

class Picture(QtGui.QWidget):
    def __init__(self):
        super(Picture, self).__init__()
        lay = QtGui.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(0)
        self.label = QLabel(self)
        self.label.setFixedSize(512, 512)
        self.label.move(10, 5)
        pic = QtGui.QPixmap("../resource/firsts.png").scaled(self.label.width(),
                                                         self.label.height())
        self.label.setPixmap(pic)
        lay.addWidget(self.label)

class Select(QtGui.QWidget):
    def __init__(self):
        super(Select, self).__init__()
        vlayout = QtGui.QVBoxLayout(self)
        vlayout.setContentsMargins(5, 10, 10, 5)
        vlayout.setSpacing(20)

        self.label = QLabel(self)
        self.label.setFixedSize(64, 64)
        picture = QtGui.QPixmap("../resource/3d.png").scaled(self.label.width(),
                                                             self.label.height())
        self.label.setPixmap(picture)

        self.texbox = QLineEdit(self)
        self.texbox.resize(250, 40)
        self.texbox.setPlaceholderText('文件路径(.nii;图片序列)')

        self.nii_button = QPushButton(self)
        self.nii_button.setText("选择nii文件")
        self.nii_button.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:gray}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}"
                                      "QPushButton{font-size:20px}")
        self.nii_button.setIcon(QIcon("../resource/open_nii.png"))
        self.nii_button.setIconSize(QtCore.QSize(20, 20))
        self.nii_button.clicked.connect(self.open_nii)

        self.pic_button = QPushButton(self)
        self.pic_button.setText("图片序列目录")
        self.pic_button.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:gray}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}"
                                      "QPushButton{font-size:20px}")
        self.pic_button.setIcon(QIcon("../resource/pic_file.png"))
        self.pic_button.setIconSize(QtCore.QSize(20, 20))
        self.pic_button.clicked.connect(self.open_pic_file)

        self.start_button = QPushButton(self)
        self.start_button.setMaximumSize(70, 70)
        self.start_button.setMinimumSize(64, 64)
        self.start_button.setStyleSheet(
            "QPushButton{background-image:url('../resource/start.png')}"
            "QPushButton{border:none;border-width:0;border-style:outset}")
        self.start_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.start_button.clicked.connect(self.start)

        self.stop_button = QPushButton(self)
        self.stop_button.setMaximumSize(70, 70)
        self.stop_button.setMinimumSize(64, 64)
        self.stop_button.setStyleSheet(
            "QPushButton{background-image:url('../resource/stop.png')}"
            "QPushButton{border:none;border-width:0;border-style:outset}")
        self.stop_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.stop_button.clicked.connect(self.stop)
        sslayout = QHBoxLayout(self)
        sslayout.setSpacing(20)
        sslayout.addWidget(self.start_button, stretch=1)
        sslayout.addWidget(self.stop_button, stretch=1)
        ssweight = QWidget()
        ssweight.setLayout(sslayout)
        vlayout.addStretch(1)
        vlayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
        vlayout.addWidget(self.texbox)
        vlayout.addWidget(self.nii_button)
        vlayout.addWidget(self.pic_button)
        vlayout.addWidget(ssweight, 0, QtCore.Qt.AlignHCenter)
        vlayout.addStretch(1)

    def open_nii(self):
        niiName, fileType = QFileDialog.getOpenFileName(self, "open nii file", "",
                                                        "*.nii;;All Files(*)")
        if not niiName[-4:] == '.nii':
            QMessageBox.information(self, "waring", "请选择nii格式的文件!",
                                    QMessageBox.Yes, QMessageBox.Cancel)
        else:
            self.texbox.setText(niiName)

    def open_pic_file(self):
        filepath = QFileDialog.getExistingDirectory(self, 'select file', './')
        self.texbox.setText(filepath)

    def start(self):
        if self.texbox.text() == None or self.texbox.text() == '':
            QMessageBox.information(self, "Tips", "请选择要处理的文件!",
                                    QMessageBox.Yes, QMessageBox.Cancel)
        else:
            wait.movie.start()
            wait.label.show()
            wait.show()
            mayavi.hide()
            pic.show()
            mayavi.visualization.scene.mlab.clf()
            model.start()
            model.signal.connect(show)

    def stop(self):
        if mayavi.isVisible():
            mayavi.hide()
            pic.show()
            mayavi.visualization.scene.mlab.clf()
        else:
            QMessageBox.information(self, "Tips", "没有可关闭的建模!",
                                    QMessageBox.Yes, QMessageBox.Cancel)

class Modeling(QThread):
    signal = pyqtSignal()

    def __init__(self):
        super(Modeling, self).__init__()

    def run(self):
        mayavi.update_data(select.texbox.text())
        self.signal.emit()

def show():
    mayavi.visualization.update_plot()
    wait.movie.stop()
    wait.label.hide()
    wait.hide()
    pic.hide()
    mayavi.show()
    print('-'*20)

class Wait(QWidget):
    def __init__(self):
        super(Wait,self).__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.resize(200,18)
        self.label = QLabel(self)
        self.label.resize(200,18)
        self.label.setWindowFlags(QtCore.Qt.FramelessWindowHint)#无边框
        self.label.setAttribute(QtCore.Qt.WA_TintedBackground)#背景透明
        self.movie = QtGui.QMovie('../resource/loading.gif')
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
    # 不创建新的QApplication，它会解开现有QApplication上由特性设置的事件。
    # 使用“.instance（）”方法来检索现有的方法。
    app = QtGui.QApplication.instance()
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap('../resource/welcomes.png').
                                     scaled(512, 512))
    splash.show()
    splash.showMessage('正在启动……')
    container = QtGui.QWidget()
    container.setWindowTitle("分割与三维重建")
    layout = QtGui.QHBoxLayout(container)
    layout.setDirection(QBoxLayout.RightToLeft)

    select = Select()
    layout.addWidget(select)
    pic = Picture()
    layout.addWidget(pic)
    mayavi = MayaviQWidget()
    mayavi.hide()
    layout.addWidget(mayavi)
    model = Modeling()
    container.show()
    wait = Wait()

    window  =QtGui.QMainWindow()
    window.resize(750, 532)
    window.setWindowTitle('CT三维重建系统')
    window.setCentralWidget(container)
    window.setWindowIcon(QIcon(os.path.dirname(os.getcwd()) + "/resource/CT.png"))
    window.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
    splash.close()
    window.show()
    app.exec_()
