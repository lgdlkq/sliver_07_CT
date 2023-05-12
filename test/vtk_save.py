#!usr/bin/env python3
# coding=utf-8
# @IDE    :PyCharm
# @File   :vtk_save.py
# @Date   :2019/5/8

from vtk.util.vtkImageImportFromArray import *
import vtk
import SimpleITK as sitk
import cv2
import numpy as np


def getSeg(source, mask):
    source = np.reshape(source,(512,512))
    mask = np.reshape(mask,(512,512))*255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masks = cv2.erode(mask, kernel)
    _, contours, _ = cv2.findContours(masks, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    for i in range(512):
        for j in range(512):
            res = cv2.pointPolygonTest(contours[0], (i, j), False)
            if res == -1:
                source[j, i] = 0.0
    return source


v_path = r'F:\sliver_07\volume-0.nii'
s_path = r'F:\sliver_07\segmentation-0.nii'  # segmentation volume
ds = sitk.ReadImage(s_path)
s_data = sitk.GetArrayFromImage(ds)
s_data[s_data == 2] = 1
s_data = s_data[40:]
s_data = s_data.astype(np.uint8)
print(s_data.shape)
# print('------------read end segmentation---------------')
# dv = sitk.ReadImage(v_path)
# v_data = sitk.GetArrayFromImage(dv)
# print('------------read end volume---------------')
# result = []
# for i in range(v_data.shape[0]):
#     if np.sum(s_data[i] == 1) < 5:
#         data = np.zeros(shape=[512, 512], dtype=np.float32)
#         result.append(data)
#         continue
#     print(i)
#     data = getSeg(v_data[i],s_data[i])
#     result.append(data)
# result = np.array(result)
# print('------------seg end---------------')
spacing = ds.GetSpacing()
print(spacing)



img_arr = vtkImageImportFromArray()
img_arr.SetArray(s_data)
img_arr.SetDataSpacing(spacing)
origin = (0, 0, 0)
img_arr.SetDataOrigin(origin)
img_arr.Update()
srange = img_arr.GetOutput().GetScalarRange()
print(srange)
print(img_arr.GetOutput().GetSpacing())
min = srange[0]
max = srange[1]
diff = max - min
slope = 4000 / diff
inter = -slope * min
shift = inter / slope

shifter = vtk.vtkImageShiftScale()  # 对偏移和比例参数来对图像数据进行操作 数据转换
shifter.SetShift(shift)
shifter.SetScale(slope)
shifter.SetOutputScalarTypeToUnsignedShort()
shifter.SetInputData(img_arr.GetOutput())
shifter.ReleaseDataFlagOff()
shifter.Update()

print(shifter.GetOutput().GetSpacing())


skinExtractor = vtk.vtkContourFilter()
skinExtractor.SetInputConnection(shifter.GetOutputPort())
skinExtractor.SetValue(0,10)

# vtkWriter = vtk.vtkXMLImageDataWriter()
vtkWriter = vtk.vtkPolyDataWriter()
vtkWriter.SetInputConnection(skinExtractor.GetOutputPort())
vtkWriter.SetFileName('../result/seg.vtk')
vtkWriter.Write()
print('------------write end---------------')

# def StartInteraction():
#     renWin.SetDesiredUpdateRate(10)
#
# def EndInteraction():
#     renWin.SetDesiredUpdateRate(0.001)
#
# def ClipVolumeRender(obj):
#     obj.GetPlanes(planes)
#     volumeMapper.SetClippingPlanes(planes)
#
# ren = vtk.vtkRenderer()
# renWin= vtk.vtkRenderWindow()
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
# volumeMapper.SetInputData(shifter.GetOutput())
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
# ren.AddVolume(newvol)
# ren.SetBackground(0, 0, 0)
# renWin.SetSize(600, 600)
#
# planes = vtk.vtkPlanes()
#
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
#
# ren.ResetCamera()
# iren.Initialize()
# renWin.Render()
# iren.Start()
