import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import onnx
from CAENet import CAENet, init_weights
from dataset import CustomImageDataset
from tqdm import tqdm

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {torch.cuda.get_device_name(device)} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # 梯度初始化置0
        loss.backward()        # 反向传播求梯度
        optimizer.step()       # 更新参数
        if batch % 8 == 0:
            with tqdm(total=size, desc='training:', postfix="loss:" + str(round(loss.item(),5)), colour='blue', ncols=120) as pbar:
                loss, current = loss.item(), batch * len(X)
                pbar.update(batch * len(X))
                # print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            min_batch_loss = loss_fn(pred, y).item()
            test_loss += min_batch_loss
            '''
            pred = pred.cpu().numpy()
            X = X.cpu().numpy()
            y = y.cpu().numpy()
            plt.matshow(pred[1, 1, :, :])
            plt.matshow(X[1, 1, :, :])
            plt.matshow(y[1, 1, :, :])
            plt.show()
            '''
            similarity = torch.dist(pred, y, 2)
            if similarity < 100:
                correct += X.shape[0]

    test_loss /= num_batches
    correct /= size
    # 存储onnx模型
    if 100 * correct > 90:
        file = "CAENet.onnx"
        im = torch.rand([1, 8, 416, 416]).to(device)
        torch.onnx.export(model, im, f=file,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        print("CAENet.onnx was saved")
        # Checks
        model_onnx = onnx.load(file)  # load onnx model
        try:
            onnx.checker.check_model(model_onnx)  # check onnx model
        except onnx.checker.ValidationError as e:
            print("model is invalid: %s" % e)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    #
    training_data = CustomImageDataset("./trainset/")
    test_data = CustomImageDataset("./testset/")
    # 超参数初始化
    batch_size = 8

    # 数据集
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y [N, C, H, W]: {y.shape}")
        break

    # 模型初始化
    model = CAENet().to(device)
    model.apply(init_weights)
    print(model)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.95, weight_decay=5e-4)
    # 学习率衰减策略
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}-------------------------------\n")
        print(f"lr:{scheduler.get_lr()}\n")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        scheduler.step()
    print("Done!")


'''
torch.onnx.export(model, x, f = "CAENet.onnx")
print(x - y)
print(y.shape)
'''