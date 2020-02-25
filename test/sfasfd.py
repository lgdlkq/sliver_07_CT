#!/usr/bin/env python

# 演示如何使用vtkboxwidget控制小部件内部的体积渲染。


import vtk
from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()
dicomReader = vtk.vtk.vtkDICOMImageReader()
dicomReader.SetDirectoryName(r'G:\PythonTrainFaile\SE2')
dicomReader.Update()
dataSpacing = dicomReader.GetDataSpacing()

# writer = vtk.vtkStructuredGridWriter()
# writer.SetInputConnection(dicomReader.GetOutputPort())
# writer.SetFileTypeToBinary()
# writer.SetFileName("SE2.vtk")
# writer.Write()

# 由于volumeMapper不能接收无符号数，所以必须加以转换
srange = dicomReader.GetOutput().GetScalarRange() #tuple
min = srange[0]
max = srange[1]

diff = max - min
slope = 4000 / diff
inter = -slope * min
shift = inter / slope

shifter = vtk.vtkImageShiftScale() #对偏移和比例参数来对图像数据进行操作  数据转换
shifter.SetShift(shift)
shifter.SetScale(slope)
shifter.SetOutputScalarTypeToUnsignedShort()
shifter.SetInputConnection(dicomReader.GetOutputPort())
shifter.ReleaseDataFlagOff()
shifter.Update()
print(min, max, slope, inter, shift)

# 加载卷，使用小部件控制渲染的卷。vtkboxwidget提供了一个剪辑体积渲染的框。

tfun = vtk.vtkPiecewiseFunction()   #梯度不透明度函数
tfun.AddPoint(2000, 0)
tfun.AddPoint(2300.0, 0.3)
tfun.AddPoint(2501.0, 1)
tfun.AddPoint(2600.0, 1)
tfun.AddPoint(2700.0, 1)
tfun.AddPoint(2900.0, 1)
tfun.AddPoint(3024.0, 1)
#tfun.AddPoint(6000.0, 1)

ctfun = vtk.vtkColorTransferFunction() #颜色传输函数
ctfun.AddRGBPoint(0.0, 0.5, 0.0, 0.0)
ctfun.AddRGBPoint(600.0, 1.0, 0.5, 0.5)
ctfun.AddRGBPoint(1280.0, 0.9, 0.2, 0.3)
ctfun.AddRGBPoint(1960.0, 0.81, 0.27, 0.1)
ctfun.AddRGBPoint(2200.0, 0.9, 0.2, 0.3)
ctfun.AddRGBPoint(2500.0, 1, 0.5, 0.5)
ctfun.AddRGBPoint(3024.0, 0.5, 0.5, 0.5)
#
#compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetInputConnection(shifter.GetOutputPort())
#volumeMapper.SetVolumeRayCastFunction(compositeFunction)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(ctfun)
volumeProperty.SetScalarOpacity(tfun)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()

newvol = vtk.vtkVolume()
newvol.SetMapper(volumeMapper)
newvol.SetProperty(volumeProperty)

outline = vtk.vtkOutlineFilter()
outline.SetInputConnection(shifter.GetOutputPort())
outlineMapper = vtk.vtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())
outlineActor = vtk.vtkActor()
outlineActor.SetMapper(outlineMapper)

# Create the RenderWindow, Renderer and both Actors
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# setInteractor方法是3D小部件如何与渲染窗口交互关联。在内部，setInteractor使用命令/观察器机制（addObserver（））设置一系列回调。
boxWidget = vtk.vtkBoxWidget()
boxWidget.SetInteractor(iren)
boxWidget.SetPlaceFactor(1.0)

# Add the actors to the renderer, set the background and size
ren.AddActor(outlineActor)
ren.AddVolume(newvol)

ren.SetBackground(0, 0, 0)
renWin.SetSize(600, 600)


# 当交互开始时，请求的帧速率会增加。
def StartInteraction(obj, event):
    global renWin
    renWin.SetDesiredUpdateRate(10)


# 当交互结束时，请求的帧速率将降低到正常水平。这将导致发生完全分辨率渲染。
def EndInteraction(obj, event):
    global renWin
    renWin.SetDesiredUpdateRate(0.001)


# 隐式函数vtkplanes与体积光线投射映射器一起使用，以限制体积的哪个部分是体积渲染的。
planes = vtk.vtkPlanes()


def ClipVolumeRender(obj, event):
    global planes, volumeMapper
    obj.GetPlanes(planes)
    volumeMapper.SetClippingPlanes(planes)


# 首先放置互动程序。读出器的输出用于放置框小部件。
#boxWidget.SetInsideOut(shifter.GetOutput())
boxWidget.PlaceWidget()
boxWidget.InsideOutOn()
boxWidget.AddObserver("StartInteractionEvent", StartInteraction)
boxWidget.AddObserver("InteractionEvent", ClipVolumeRender)
boxWidget.AddObserver("EndInteractionEvent", EndInteraction)

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

iren.Initialize()
renWin.Render()
iren.Start()


# filename="E:\All Files\medical image\Display-3D-GFR\GFR3d\SHEN_YUN_WANG\SE2\out.png"
#
# windowToImageFilter = vtk.vtkWindowToImageFilter()
# windowToImageFilter.SetInput(renWin)
# windowToImageFilter.Update()
# writer = vtk.vtkPNGWriter()
# writer.SetFileName(filename)
# writer.SetInputData(windowToImageFilter.GetOutputPort())
# writer.Write()