#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :test.py
# @Date   :2019/5/3

from vtk.util.vtkImageImportFromArray import *
import vtk
import SimpleITK as sitk
import numpy as np
import cv2

path = r'F:\sliver_07\segmentation-0.nii' #segmentation volume
ds = sitk.ReadImage(path)
data = sitk.GetArrayFromImage(ds)
spacing = ds.GetSpacing()
data = data[46:]
data = data.astype(np.uint8)

# n_data = []
# length = int(spacing[2])-1
# spacing = (spacing[0],spacing[1],1.0)
# for i in range(data.shape[0]-1):
#     n_data.append(data[i])
#     for j in range(length):
#         temp = cv2.resize(data[i],(512+10*(j+1),512+10*(j+1)))
#         temp = temp[5*(j+1):-5*(j+1),5*(j+1):-5*(j+1)]
#         n_data.append(temp)
# n_data = np.array(n_data)
#

img_arr = vtkImageImportFromArray()
img_arr.SetArray(data)
img_arr.SetDataSpacing(spacing)
origin = (0,0,0)
img_arr.SetDataOrigin(origin)
img_arr.Update()
srange = img_arr.GetOutput().GetScalarRange()

print('spacing: ',spacing)
print('srange: ',srange)

# def StartInteraction():
#     renWin.SetDesiredUpdateRate(10)
#
# def EndInteraction():
#     renWin.SetDesiredUpdateRate(0.001)
#
# def ClipVolumeRender(obj):
#     obj.GetPlanes(planes)
#     volumeMapper.SetClippingPlanes(planes)

ren = vtk.vtkRenderer()
renWin= vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

min = srange[0]
max = srange[1]
diff = max - min
slope = 4000 / diff
inter = -slope * min
shift = inter / slope
print(min, max, slope, inter, shift)

shifter = vtk.vtkImageShiftScale()  # 对偏移和比例参数来对图像数据进行操作 数据转换
shifter.SetShift(shift)
shifter.SetScale(slope)
shifter.SetOutputScalarTypeToUnsignedShort()
shifter.SetInputData(img_arr.GetOutput())#************************
shifter.ReleaseDataFlagOff()
shifter.Update()


# diffusion = vtk.vtkImageAnisotropicDiffusion3D()
# diffusion.SetInputData(shifter.GetOutput())
# diffusion.SetNumberOfIterations(10)
# diffusion.SetDiffusionThreshold(1)
# diffusion.Update()

tfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数
tfun.AddPoint(1000.0, 0)
tfun.AddPoint(2300.0, 0.3)
tfun.AddPoint(2501.0, 0.4)
tfun.AddPoint(2600.0, 0.5)
tfun.AddPoint(2700.0, 0.6)
tfun.AddPoint(2900.0, 0.7)
tfun.AddPoint(3024.0, 1)

ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数
ctfun.AddRGBPoint(0.0, 0.5, 0.0, 0.0)
ctfun.AddRGBPoint(600.0, 1.0, 0.5, 0.5)
ctfun.AddRGBPoint(1280.0, 0.9, 0.2, 0.3)
ctfun.AddRGBPoint(1960.0, 0.81, 0.27, 0.1)
ctfun.AddRGBPoint(2200.0, 0.9, 0.2, 0.3)
ctfun.AddRGBPoint(2500.0, 1, 0.5, 0.5)
ctfun.AddRGBPoint(3024.0, 0.5, 0.5, 0.5)

volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetInputData(shifter.GetOutput())#*************
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(ctfun)
volumeProperty.SetScalarOpacity(tfun)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()

newvol = vtk.vtkVolume()
newvol.SetMapper(volumeMapper)
newvol.SetProperty(volumeProperty)

# outline = vtk.vtkOutlineFilter()
# outline.SetInputConnection(shifter.GetOutputPort())
#
# outlineMapper = vtk.vtkPolyDataMapper()
# outlineMapper.SetInputConnection(outline.GetOutputPort())
#
# outlineActor = vtk.vtkActor()
# outlineActor.SetMapper(outlineMapper)
#
# ren.AddActor(outlineActor)
ren.AddVolume(newvol)
ren.SetBackground(0, 0, 0)
renWin.SetSize(600, 600)

# planes = vtk.vtkPlanes()

# boxWidget = vtk.vtkBoxWidget()
# boxWidget.SetInteractor(iren)
# boxWidget.SetPlaceFactor(1.0)
# boxWidget.PlaceWidget(0,0,0,0,0,0)
# boxWidget.InsideOutOn()
# boxWidget.AddObserver("StartInteractionEvent", StartInteraction)
# boxWidget.AddObserver("InteractionEvent",  ClipVolumeRender)
# boxWidget.AddObserver("EndInteractionEvent",  EndInteraction)
#
# outlineProperty = boxWidget.GetOutlineProperty()
# outlineProperty.SetRepresentationToWireframe()
# outlineProperty.SetAmbient(1.0)
# outlineProperty.SetAmbientColor(1, 1, 1)
# outlineProperty.SetLineWidth(9)
#
# selectedOutlineProperty = boxWidget.GetSelectedOutlineProperty()
# selectedOutlineProperty.SetRepresentationToWireframe()
# selectedOutlineProperty.SetAmbient(1.0)
# selectedOutlineProperty.SetAmbientColor(1, 0, 0)
# selectedOutlineProperty.SetLineWidth(3)

ren.ResetCamera()
iren.Initialize()
renWin.Render()
iren.Start()
