"""
这个例子演示了使用mayavi作为大qt的一个组件应用。mayavi被嵌入到一个qwidget中。
"""
# 在导入任何包之前，将ets_toolkit环境变量设置为qt4，以告诉traits使用qt。
import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

os.environ['ETS_TOOLKIT'] = 'qt4'
# 默认情况下，将使用Pyside绑定。如果使用pyqt绑定，需将qt_api环境变量设置为“pyqt”
# os.environ['QT_API'] = 'pyqt'

# 为了能使用pyside或pyqt4，并且不会与特性冲突。我们需从pyface.qt导入qtgui和qtcore
from pyface.qt import QtGui, QtCore
# 或者，可以绕过这一行，但要确保在导入pyqt之前执行以下行
#   import sip
#   sip.setapi('QString', 2)

################################################################################

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
import SimpleITK as sitk


################################################################################
# The actual visualization 当前的视图
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        # 打开视图时调用此函数。视图尚未打开时不会填充场景，因某些VTK功能需GLContext
        # 可以在嵌入的场景上进行正常的MLAB调用
        self.scene.mlab
        path = r'F:\sliver_07\segmentation-0.nii'  # segmentation-0.nii  volume-0.nii
        ds = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(ds)
        self.scene.mlab.contour3d(image, color=(0.53, 0, 0.098),
                                  transparent=False, opacity=1.0)

        # self.scene.mlab.test_points3d()

    #创建对话框布局
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=512, width=512, show_label=False),
                resizable=False  # 需要用父窗口小部件调整大小
                )


################################################################################
# 包含可视化的qwidget，pyqt4代码
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # 如果要调试，需要删除qt输入挂钩。
        # QtCore.pyqtRemoveInputHook()
        # import pdb
        # pdb.set_trace()
        # QtCore.pyqtRestoreInputHook()

        # 编辑特性调用将生成要嵌入的小部件
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


if __name__ == "__main__":
    # 不创建新的QApplication，它解开现有QApplication上由特性设置的事件。使用“.instance（）”方法来检索现有的方法。
    app = QtGui.QApplication.instance()
    container = QtGui.QWidget()
    container.setWindowTitle("reconstruction")
    # 定义“复杂”布局以测试行为
    layout = QtGui.QGridLayout(container)
    path = r'F:\sliver_07\segmentation-0.nii'
    # 在mayavi周围放些东西
    # label_list = []
    # for i in range(3):
    #     for j in range(3):
    #         if (i==1) and (j==1):continue
    #         label = QtGui.QLabel(container)
    #         label.setText("Your QWidget at (%d, %d)" % (i,j))
    #         label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
    #         layout.addWidget(label, i, j)
    #         label_list.append(label)
    mayavi_widget = MayaviQWidget(container)

    layout.addWidget(mayavi_widget, 1, 1)
    container.show()
    window = QtGui.QMainWindow()
    window.setWindowTitle('CT三维重建系统')
    window.setCentralWidget(container)
    window.show()
    # 启动主事件循环。
    app.exec_()
