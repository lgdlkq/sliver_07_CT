#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :vtk_ct.py
# @Date   :2019/4/26


import vtk
from PyQt5 import  QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys
from PyQt5.QtWidgets import QApplication
from vtk.util.misc import vtkGetDataRoot

class myMainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.frame = QtWidgets.QFrame()

        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.renWin= self.vtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = self.renWin.GetInteractor()
        self.iren.SetRenderWindow(self.renWin)

        dicomReader = vtk.vtk.vtkDICOMImageReader()
        dicomReader.SetDirectoryName(r'G:\PythonTrainFaile\SE2')#G:\PythonTrainFaile\SE2  F:\sliver_07\segmentation-0.nii
        dicomReader.Update()
        srange = dicomReader.GetOutput().GetScalarRange()
        print(dicomReader.GetOutput().GetDimensions())
        min = srange[0]
        max = srange[1]
        diff = max - min
        slope = 4000 / diff
        inter = -slope * min
        shift = inter / slope

        shifter = vtk.vtkImageShiftScale()  # 对偏移和比例参数来对图像数据进行操作  数据转换
        shifter.SetShift(shift)
        shifter.SetScale(slope)
        shifter.SetOutputScalarTypeToUnsignedShort()
        shifter.SetInputData(dicomReader.GetOutput())
        shifter.ReleaseDataFlagOff()
        shifter.Update()
        print(min, max, slope, inter, shift)
        tfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数
        tfun.AddPoint(2000, 0)
        tfun.AddPoint(2300.0, 0.3)
        tfun.AddPoint(2501.0, 1)
        tfun.AddPoint(2600.0, 1)
        tfun.AddPoint(2700.0, 1)
        tfun.AddPoint(2900.0, 1)
        tfun.AddPoint(3024.0, 1)

        ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数
        ctfun.AddRGBPoint(0.0, 0.5, 0.0, 0.0)
        ctfun.AddRGBPoint(600.0, 1.0, 0.5, 0.5)
        ctfun.AddRGBPoint(1280.0, 0.9, 0.2, 0.3)
        ctfun.AddRGBPoint(1960.0, 0.81, 0.27, 0.1)
        ctfun.AddRGBPoint(2200.0, 0.9, 0.2, 0.3)
        ctfun.AddRGBPoint(2500.0, 1, 0.5, 0.5)
        ctfun.AddRGBPoint(3024.0, 0.5, 0.5, 0.5)
        self.volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        self.volumeMapper.SetInputConnection(shifter.GetOutputPort())
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(ctfun)
        volumeProperty.SetScalarOpacity(tfun)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        newvol = vtk.vtkVolume()
        newvol.SetMapper(self.volumeMapper)
        newvol.SetProperty(volumeProperty)
        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(shifter.GetOutputPort())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)

        # Create an actor
        boxWidget = vtk.vtkBoxWidget()
        boxWidget.SetInteractor(self.iren)
        boxWidget.SetPlaceFactor(1.0)

        # Add the actors to the renderer, set the background and size
        self.ren.AddActor(outlineActor)
        self.ren.AddVolume(newvol)

        self.ren.SetBackground(0, 0, 0)
        self.renWin.SetSize(600, 600)

        self.planes = vtk.vtkPlanes()

        boxWidget.PlaceWidget()
        boxWidget.InsideOutOn()
        boxWidget.AddObserver("StartInteractionEvent", self.StartInteraction)
        boxWidget.AddObserver("InteractionEvent",  self.ClipVolumeRender)
        boxWidget.AddObserver("EndInteractionEvent",  self.EndInteraction)

        outlineProperty = boxWidget.GetOutlineProperty()
        outlineProperty.SetRepresentationToWireframe()
        outlineProperty.SetAmbient(1.0)
        outlineProperty.SetAmbientColor(1, 1, 1)
        outlineProperty.SetLineWidth(9)

        selectedOutlineProperty = boxWidget.GetSelectedOutlineProperty()
        selectedOutlineProperty.SetRepresentationToWireframe()
        selectedOutlineProperty.SetAmbient(1.0)
        selectedOutlineProperty.SetAmbientColor(1, 0, 0)
        selectedOutlineProperty.SetLineWidth(3)

        self.ren.ResetCamera()
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def StartInteraction(self,obj, event):
        self.renWin.SetDesiredUpdateRate(10)

    def EndInteraction(self,obj, event):
        self.renWin.SetDesiredUpdateRate(0.001)

    def ClipVolumeRender(self,obj, event):
        obj.GetPlanes(self.planes)
        self.volumeMapper.SetClippingPlanes(self.planes)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = myMainWindow()

    sys.exit(app.exec_())
