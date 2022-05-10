# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(6, 128)
      self.fc2 = nn.Linear(128, 128)
      self.fc3 = nn.Linear(128, 2)
      self.dropout = nn.Dropout(0.5)

    # x represents our data
    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      # x = self.dropout(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      # Apply softmax to x
      # output = F.log_softmax(x, -1)
      output = F.softmax(x, -1)
      return output


def load(file):
    data = torch.from_numpy(np.load(file)).float()
    data = torch.where(torch.isnan(data), torch.full_like(data, 0), data)
    return data

def train():
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # # test
    # test_out = net(combine_model_train_X[0])
    # print(test_out, better_model_train_Y[0])
    # print(criterion(test_out, better_model_train_Y[0].long()))
    for epoch in range(100):
        # output = model(combine_model_train_X)
        # loss = criterion(output, better_model_train_Y.long())
        output = model(all_X)
        loss = criterion(output, all_Y.long())

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # better_model_train_Y_pred = torch.argmax(output, dim=1)
        # acc1 = accuracy_score(better_model_train_Y_pred, better_model_train_Y)
        # better_model_test_Y_pred = torch.argmax(model(combine_model_test_X), dim=1)
        # acc2 = accuracy_score(better_model_test_Y_pred, better_model_test_Y)
        all_Y_pred = torch.argmax(output, dim=1)
        acc3 = accuracy_score(all_Y, all_Y_pred)
        print('Epoch: [{}], Loss: {}, All acc: {}'.format(epoch, loss.item(), acc3))

        # print('Epoch: [{}], Loss: {}, Train acc: {}, Test acc: {}'.format(epoch, loss.item(), acc1, acc2))


if __name__ == '__main__':
    model = Net()
    # print(model)
    combine_model_train_X = load('combine_model_train_X.npy')
    combine_model_test_X = load('combine_model_pred_X.npy')
    better_model_train_Y = load('better_model_train_Y.npy')
    better_model_test_Y = load('tools/better_model_test_Y.npy')
    print(np.shape(combine_model_train_X))

    device = torch.device('cuda:0')
    combine_model_train_X.to(device)
    combine_model_test_X.to(device)
    better_model_train_Y.to(device)
    better_model_test_Y.to(device)
    # # model.to(device)

    all_X = torch.cat((combine_model_train_X, combine_model_test_X), 0)
    all_Y = torch.cat((better_model_train_Y, better_model_test_Y), 0)

    train()
    # better_model_test_Y_pred = torch.argmax(model(combine_model_test_X), dim=1)
    # acc = accuracy_score(better_model_test_Y_pred, better_model_test_Y)
    # print(acc)