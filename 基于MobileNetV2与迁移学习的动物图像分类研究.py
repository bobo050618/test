import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 数据增强预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据集加载
# 数据集本地路径
# 数据集自己到网上下载
data_root = r"D:\ANIMALS\ANIMALS\raw-img"

# 自动读取10个类别文件夹
dataset = ImageFolder(root=data_root, transform=transform_train)
class_names = dataset.classes
print("成功加载数据集！类别列表：", class_names)
print("总图片数量：", len(dataset))

# 划分训练集80% / 测试集20%
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# MobileNetV2 迁移学习
model = models.mobilenet_v2(pretrained=True)
# 冻结底层特征提取层
for param in model.features.parameters():
    param.requires_grad = False
# 替换分类头，适配10类动物
model.classifier[1] = nn.Linear(model.last_channel, 10)

# 自动使用GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("模型加载完成，使用设备：", device)

# 训练超参数配置
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)
epochs = 20

# 保存训练曲线数据
train_loss_hist, train_acc_hist = [], []
test_loss_hist, test_acc_hist = [], []

# 完整训练循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    train_loss /= train_total
    train_acc = 100 * train_correct / train_total

    # 测试验证阶段
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

    test_loss /= test_total
    test_acc = 100 * test_correct / test_total

    # 保存历史数据
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)

    scheduler.step()

    # 打印每轮训练结果
    print(f"Epoch [{epoch + 1:02d}/{epochs}] | "
          f"训练损失: {train_loss:.4f} | "
          f"测试损失: {test_loss:.4f} | "
          f"训练准确率: {train_acc:.2f}% | "
          f"测试准确率: {test_acc:.2f}%")

# 训练曲线可视化
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决matplotlib中文乱码
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_acc_hist, label="训练集准确率")
plt.plot(test_acc_hist, label="测试集准确率")
plt.title("模型准确率变化曲线")
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_hist, label="训练集损失")
plt.plot(test_loss_hist, label="测试集损失")
plt.title("模型损失变化曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ===================== 7. 补充：可视化相关代=====================
# 7.1 特征图可视化
def visualize_feature_map():
    import matplotlib.pyplot as plt
    # 选取一张测试集图像
    img, label = next(iter(test_loader))
    img = img[0].unsqueeze(0).to(device)

    # 定义钩子函数，提取特征图
    features = []

    def hook_fn(module, input, output):
        features.append(output.detach().cpu())

    # 注册钩子，提取MobileNetV2第一层卷积的特征图
    handle = model.features[0].register_forward_hook(hook_fn)

    # 前向传播，获取特征图
    model.eval()
    with torch.no_grad():
        model(img)

    # 取消钩子
    handle.remove()

    # 绘制特征图（展示前8个通道）
    plt.figure(figsize=(12, 4))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(features[0][0][i], cmap='gray')
        plt.axis('off')
    plt.title("MobileNetV2第一层卷积特征图")
    plt.show()


# 7.2 t-SNE特征降维可视化
def visualize_tsne():
    from sklearn.manifold import TSNE
    import numpy as np
    # 提取测试集特征和标签
    features_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # 提取全连接层前的特征
            features = model.features(images).mean(dim=[2, 3])
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(labels.numpy())

    # 合并特征和标签
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 绘制t-SNE图
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for i in range(10):
        mask = labels == i
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                    c=[colors[i]], label=class_names[i], alpha=0.7)
    plt.legend()
    plt.title("t-SNE特征降维可视化（10类动物）")
    plt.show()


# 7.3 注意力图可视化（Grad-CAM）
def visualize_attention_map():
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import cv2
    # 选取目标层
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layer=target_layer)

    # 选取一张测试集图像
    img, label = next(iter(test_loader))
    img_np = img[0].permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # 生成注意力图
    input_tensor = img[0].unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # 绘制注意力图
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("原始图像")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("注意力图（模型关注区域）")
    plt.axis('off')
    plt.show()


# 调用可视化函数
visualize_feature_map()
visualize_tsne()
visualize_attention_map()
