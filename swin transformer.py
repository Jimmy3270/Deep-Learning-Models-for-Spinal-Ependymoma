import os
import random
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import cv2
import paddleseg.transforms as T
from paddleseg.datasets import OpticDiscSeg,Dataset

# 建立transforms,进行数据增强
train_transforms = [
    T.RandomHorizontalFlip(),                                                              # 水平翻转
    T.RandomDistort(),                                                                     # 随机扭曲
    T.RandomRotation(max_rotation = 10,im_padding_value =(0,0,0),label_padding_value = 0), # 随机旋转
    T.RandomBlur(),                                                                        # 随机模糊
    T.RandomScaleAspect(min_scale = 0.8, aspect_ratio = 0.5),                              # 随机缩放
    T.Resize(target_size=(225, 225)),
    T.Normalize()                                                                          # 归一化 mean Default: [0.5, 0.5, 0.5]  std Default: [0.5, 0.5, 0.5].
]

val_transforms = [
    T.Resize(target_size=(225, 225)),
    T.Normalize()
]

test_transforms = [
    T.Resize(target_size=(225, 225)),
    T.Normalize()
]

dataset_root = 'D:/projectST/newdata'
train_path  = 'D:/projectST/newdata/train_list.txt'
val_path  = 'D:/projectST/newdata/val_list.txt'
test_path  = 'D:/projectST/newdata/test_list.txt'

# 构建训练集
train_dataset = Dataset(
    dataset_root=dataset_root,
    train_path=train_path,
    transforms=train_transforms,
    num_classes=2,
    mode='train'
    )

# 构建验证集
val_dataset = Dataset(
    dataset_root=dataset_root,
    val_path=val_path,
    transforms=val_transforms,
    num_classes=2,
    mode='val'
    )

# 构建测试集
test_dataset = Dataset(
    dataset_root=dataset_root,
    test_path=test_path,
    transforms=test_transforms,
    num_classes=2,
    mode='test'
    )


#显示 img与label
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(16,16))
for i in range(1,6,2):
    img, label = train_dataset[100]
    img = np.transpose(img, (1,2,0))
    img = img*0.5 + 0.5
    plt.subplot(3,2,i),plt.imshow(img,'gray'),plt.title('img'),plt.xticks([]),plt.yticks([])
    plt.subplot(3,2,i+1),plt.imshow(label,'gray'),plt.title('label'),plt.xticks([]),plt.yticks([])
    #plt.show


#导入模型
from paddleseg.models import MLATransformer
from paddleseg.models import backbones
from paddleseg.cvlibs import manager

#backbone=backbones.SwinTransformer_small_patch4_window7_224
#backbone.feat_channels
num_classes = 2
pret=r"https://bj.bcebos.com/paddleseg/dygraph/swin_transformer_base_patch4_window7_224_imagenet_1k.tar.gz"
model = MLATransformer(
    num_classes=2,
    backbone=manager.BACKBONES['SwinTransformer_base_patch4_window7_224'](),
    in_channels=[128,256,512,1024],
    pretrained=pret,
)
print(model)

# 模型参数设置
from paddleseg.models.losses import CrossEntropyLoss,DiceLoss
import paddle
# gpu计算
print(paddle.device.get_device())
#paddle.device.set_device('cpu')
# 设置学习率
base_lr = 0.005
lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=1800, verbose=False)
# 设置优化器
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
# 组合dice损失函数
losses = {}
losses['types'] = [CrossEntropyLoss()]*2
losses['coef'] = [1]*2

# 模型训练
from paddleseg.core import train

train(
    model=model,
    train_dataset=train_dataset,       # 填写训练集的dataset
    val_dataset=val_dataset,           # 填写验证集的dataset
    optimizer=optimizer,               # 优化器
    save_dir='D:/projectST/seg model',    # 保存路径
    iters=100000,                        # 训练次数
    batch_size=4,                      # 每批处理图片的张数
    save_interval=200,                 # 保存的间隔次数
    log_iters=10,                      # 日志打印间隔
    num_workers=0,                     # 异步加载数据的进程数目
    losses=losses,
               # 传入loss函数
    )                      # 是否使用visualDL


# 模型评估（验证集）
import paddle
from paddleseg.core import evaluate


model =  MLATransformer(
    num_classes=2,
    backbone=manager.BACKBONES['SwinTransformer_base_patch4_window7_224'](),
    in_channels=[128,256,512,1024],
    pretrained=pret
)
# 换自己保存的模型文件
model_path = 'D:/projectST/seg model/best_model/model.pdparams'
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)
evaluate(model,val_dataset)


# 模型测试（测试集）与保存
from paddleseg.core import predict
transforms = T.Compose([
    T.Resize(target_size=(225, 225)),
    T.Normalize()
])

model =  MLATransformer(
    num_classes=2,
    backbone=manager.BACKBONES['SwinTransformer_base_patch4_window7_224'](),
    in_channels=[128,256,512,1024],
    pretrained=pret
)
# 生成图片列表
image_list = []
with open('D:/projectST/newdata/test_list.txt' ,'r') as f:
    for line in f.readlines():
        image_list.append(line.split()[0])

predict(
        model,
        # 保存的模型文件
        model_path = 'D:/projectST/seg model/best_model/model.pdparams',
        transforms=transforms,
        image_list=image_list,
        save_dir='D:/projectST/seg model/results',
    )

# 显示部分预测结果，并将其与label对比
num = 4
img_list = random.sample(image_list, num)
pre_path = 'D:/projectST/seg model/results/pseudo_color_prediction'
plt.figure(figsize=(12,num*2))
index = 1
for i in range(len(img_list)):
    plt.subplot(num,3,index)
    img_origin = cv2.imread(img_list[i],0)
    plt.title('origin')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_origin,'gray')

    plt.subplot(num,3,index+1)
    label_path = (img_list[i].replace('origin', 'label')).replace('jpg','png')
    img_label = cv2.imread(label_path,0)
    plt.title('label')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_label, 'gray')

    plt.subplot(num,3,index+2)
    predict_path = os.path.join(pre_path, os.path.basename(label_path))
    img_pre = cv2.imread(predict_path,0)
    plt.title('predict')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_pre, 'gray')

    index += 3

plt.show()









