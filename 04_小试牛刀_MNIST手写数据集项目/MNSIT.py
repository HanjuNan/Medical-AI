import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
tranformer = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST(root="./data",train=True,transform=tranformer,download=True)
test_data = torchvision.datasets.MNIST(root="./data",train=False,transform=tranformer,download=True)


dl_train = torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
dl_test = torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=True)

print("dl_train = ",dl_train)
print("dl_test = ",dl_test)


image,label = next(iter(dl_train))
print("image.shape = ",image.shape)
print("label.shape = ",label.shape)

# 取一张看看
im = image[0]
im = im.numpy()
print("im.shape = ",im.shape)

# 与维度为3不同，维度为1需要直接去掉
im = im.squeeze()
print("im.shape = ",im.shape)
# plt.imshow(im)
# plt.show()

plt.figure(figsize=(16,8))
for i in range(len(image[:8])):
    img = image[:8][i]
    img = img.numpy()
    img = img.squeeze()
    label_img = label[:8][i].numpy()
    plt.subplot(2,4,i+1)
    plt.title(label_img)
    plt.imshow(img)
plt.show()










