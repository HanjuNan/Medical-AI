import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# array(读取图片和画图) tensor(用于模型预测) ## 读取数据np--tensor--model--result--np--画
np1 = np.array([[1,2,3,4,5,6],[2,3,4,5,6,7]])
print("np1 = ",np1)

# numpy转Tensor
tensor1 = torch.from_numpy(np1)
print("tensor1 = ",tensor1)

# tensor转cuda
tensor1 = tensor1.to("cuda")
print("tensor1 = ",tensor1)

# cuda从int32转float32
tensor1 = tensor1.type(torch.float32)
print("tensor1 = ",tensor1)

# cuda转cpu
tensor1 = tensor1.cpu()
print("tensor1 = ",tensor1)

# tensor转numpy
tensor1 = tensor1.numpy()
print("tensor1 = ",tensor1)

# 查看shape
print(tensor1.shape)


# 读取图片和查看图片形状
path = "./1.jpg"
pic = Image.open(fp=path)
# pic =  <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=515x300 at 0x19CE1EFF978>
print("pic = ",pic)
print("type(pic) = ",type(pic))
pic = np.array(pic)
# print("pic = ",pic)
print("type(pic) = ",type(pic))
print(pic.shape)

# 将numpy转为Tensor
pic_tensor = torch.from_numpy(pic)
print("pic_tensor = ",pic_tensor.shape)

# 显示图片
path = "./1.jpg"
pic = Image.open(fp=path)
plt.imshow(pic)
plt.show()





















