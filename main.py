import os
import argparse

from mindspore import context, nn, Model
from mindspore.nn import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from net import LeNet5
from params import train_epoch, mnist_path, dataset_size
from train import train_net, test_net


def parse_arguments():
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    args = parser.parse_known_args()[0]
    print(args.device_target)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)


def main():
    # 设置运行方式
    parse_arguments()

    # 实例化网络
    net = LeNet5()

    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义优化器
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    # 模型
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
    test_net(model, mnist_path)


if __name__ == '__main__':
    main()
