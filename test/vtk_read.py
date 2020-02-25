#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :vtk_read.py
# @author :雷国栋
# @Date   :2019/5/8

import vtk
renderer = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

vtkReader=vtk.vtkPolyDataReader()
vtkReader.SetFileName(r"../result/seg.vtk")
vtkReader.Update()
srange = vtkReader.GetOutput().GetScalarRange()
print(srange)

skinMapper = vtk.vtkPolyDataMapper()
skinMapper.SetInputData(vtkReader.GetOutput())
skinMapper.ScalarVisibilityOff()

skinActor = vtk.vtkActor()
skinActor.SetMapper(skinMapper)
skinActor.GetProperty().SetInterpolationToGouraud()

renderer.AddActor(skinActor)
renWin.Render()
iren.Initialize()
iren.Start()

# def StartInteraction():
#     renWin.SetDesiredUpdateRate(10)
#
# def EndInteraction():
#     renWin.SetDesiredUpdateRate(0.001)
#
# def ClipVolumeRender(obj):
#     obj.GetPlanes(planes)
#     volumeMapper.SetClippingPlanes(planes)

# ren = vtk.vtkRenderer()
# renWin=vtk.vtkRenderWindow()
# iren=vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
# renWin.AddRenderer(ren)
# iren = vtk.vtkRenderWindowInteractor()
# iren.SetRenderWindow(renWin)
#
# tfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数
# tfun.AddPoint(1000.0, 0)
# tfun.AddPoint(2300.0, 0.3)
# tfun.AddPoint(2501.0, 0.4)
# tfun.AddPoint(2600.0, 0.5)
# tfun.AddPoint(2700.0, 0.6)
# tfun.AddPoint(2900.0, 0.7)
# tfun.AddPoint(3024.0, 1)
#
#
# ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数
# ctfun.AddRGBPoint(0.0, 0.5, 0.0, 0.0)
# ctfun.AddRGBPoint(600.0, 1.0, 0.5, 0.5)
# ctfun.AddRGBPoint(1280.0, 0.9, 0.2, 0.3)
# ctfun.AddRGBPoint(1960.0, 0.81, 0.27, 0.1)
# ctfun.AddRGBPoint(2200.0, 0.9, 0.2, 0.3)
# ctfun.AddRGBPoint(2500.0, 1, 0.5, 0.5)
# ctfun.AddRGBPoint(3024.0, 0.5, 0.5, 0.5)
#
# volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
# volumeMapper.SetInputData(vtkReader.GetOutput())
# volumeProperty = vtk.vtkVolumeProperty()
# volumeProperty.SetColor(ctfun)
# volumeProperty.SetScalarOpacity(tfun)
# volumeProperty.SetInterpolationTypeToLinear()
# volumeProperty.ShadeOn()
#
# newvol = vtk.vtkVolume()
# newvol.SetMapper(volumeMapper)
# newvol.SetProperty(volumeProperty)
#
# ren.AddVolume(newvol)
# ren.SetBackground(0, 0, 0)
# renWin.SetSize(600, 600)
#
# planes = vtk.vtkPlanes()
#
# ren.ResetCamera()
# iren.Initialize()
# renWin.Render()
# iren.Start()


