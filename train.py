from torch.utils.tensorboard import SummaryWriter

from model import *
import torchvision
from torch import nn


# 准备数据集
from torch.utils.data import DataLoader

train_data =torchvision.datasets.CIFAR10("root=./data",train=True,transform=torchvision.transforms.ToTensor()
                                         ,download=True)
test_data = torchvision.datasets.CIFAR10("root=./data",train=False,transform=torchvision.transforms.ToTensor()
                                         ,download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)

# 如果train_data_size=10，则打印“训练数据集长度为10”
print("训练数据集的长度为：{}".format(train_data_size))
print("训练训练集的长度为：{}".format(test_data_size))

# 利用 DataLoader来记载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
hp = Hp()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(hp.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
# 添加TensorBoard
writer = SummaryWriter("./logs_train")
for i in range(epoch):
    print("-----------------------第{}轮训练开始---------------".format(i+1))
    # 训练步骤开始
    # hp.train()
    for data in train_dataloader:
        imgs , targets = data
        outputs = hp (imgs)
        loss =loss_fn(outputs,targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step +1
        if total_train_step % 100 ==0:
            print("训练次数：{}，loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
  # 测试步骤开始
  # hp.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = hp(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()

            # 计算整体测试集正确率 = 整体对的个数 / 整体的个数
            # 难点
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy


    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))

    # 增加损失值及正确率到tensorboard中
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)


    total_test_step =total_test_step + 1

    # 保存模型
    torch.save(hp,"hp_{}.pth".format(i))
    # torch.save(hp.state_dict(),"hp_{}.pth".format(i))
    print("模型已保存")
writer.close()

# this is finish 


