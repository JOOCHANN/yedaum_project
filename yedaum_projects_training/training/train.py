from data_loader import load
import torch
from train_model import Text_CNN
from torch import nn
import time
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

batch_size = 128
learning_rate = 0.0001
num_epoch = 20
learing_rate_decay = 10
data_path = '../data/ilbe_Crawling_200000.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu 사용

#evaluation function
def eval(x_loader,y_loader, model_name):
    correct = 0
    total = 0

    with torch.no_grad():
        for j, (data, label) in enumerate(zip(x_loader,y_loader)):
            x = data.to(device)
            y_ = label.to(device)

            output = model_name.forward(x)

            for i in range(0, len(output)):
                if(output[i] > 0.5):
                    output[i] = 1
                else:
                    output[i] = 0

            output = output.view(len(output))

            total += label.size(0)
            correct += (output == y_).sum().float()
    return 100*correct/total


# f1_score function
def f1_score(x_loader, y_loader, model_name, device):
    with torch.no_grad():
        for j, (data, label) in enumerate(zip(x_loader, y_loader)):
            x = data.to(device)
            y_ = label.to(device)

            output = model_name.forward(x)

            for i in range(0, len(output)):
                if (output[i] >= 0.5):
                    output[i] = 1
                else:
                    output[i] = 0

            # y_true[j] = output
            if j == 0:
                y_true = y_
                y_pred = output
            else:
                y_true = torch.cat([y_true, y_])
                y_pred = torch.cat([y_pred, output])

        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        y_true = y_true.tolist()
        y_pred = y_pred.tolist()

        target_names = ['class 0', 'class 1']
        print(classification_report(y_true, y_pred, target_names=target_names))

def main():
    print("data loading...")
    loading = load(data_path)
    train_size, test_size, val_size = loading.return_len()

    x_train_torch = torch.empty(train_size, 127, 200)
    x_val_torch = torch.empty(test_size, 127, 200)
    x_test_torch = torch.empty(val_size, 127, 200)
    y_train_torch = torch.empty(train_size)
    y_val_torch = torch.empty(test_size)
    y_test_torch = torch.empty(val_size)

    print(x_train_torch.shape)
    print(x_test_torch.shape)
    print(x_val_torch.shape)

    x_train_torch, x_val_torch, x_test_torch, y_train_torch, y_val_torch, y_test_torch = loading.main_processing()

    print(x_train_torch.shape, x_val_torch.shape, x_test_torch.shape, y_train_torch.shape, y_val_torch.shape, y_test_torch.shape)

    print("data loading success")

    x_train_loader = torch.utils.data.DataLoader(x_train_torch, batch_size=batch_size
                                                 , shuffle=False, num_workers=0, drop_last=True)
    y_train_loader = torch.utils.data.DataLoader(y_train_torch, batch_size=batch_size
                                                 , shuffle=False, num_workers=0, drop_last=True)

    x_val_loader = torch.utils.data.DataLoader(x_val_torch, batch_size=batch_size
                                               , shuffle=False, num_workers=0, drop_last=True)
    y_val_loader = torch.utils.data.DataLoader(y_val_torch, batch_size=batch_size
                                               , shuffle=False, num_workers=0, drop_last=True)

    x_test_loader = torch.utils.data.DataLoader(x_test_torch, batch_size=batch_size
                                                , shuffle=False, num_workers=0, drop_last=True)
    y_test_loader = torch.utils.data.DataLoader(y_test_torch, batch_size=batch_size
                                                , shuffle=False, num_workers=0, drop_last=True)
    model = Text_CNN().to(device)  # 모델을 gpu에 올림
    loss_func = nn.BCELoss()  # lossfunction 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer 정의

    loss_arr = [[0 for i in range(len(x_train_loader))] for j in range(num_epoch)]
    train_acc = []
    val_acc = []
    for i in range(num_epoch):
        start = time.time()  # 시간 측정
        for j, (data, label) in enumerate(zip(x_train_loader, y_train_loader)):

            x = data.to(device)  # data를 gpu에 올림
            y_ = label.to(device)  # label을 gpu에 올림

            optimizer.zero_grad()  # optimizer 초기화
            output = model.forward(x)  # 모델 foward 진행
            loss = loss_func(output, y_)  # loss function을 사용해서 loss 측정.
            loss.backward()  # 가중치에 대한 Loss의 변화량을 측정함.
            optimizer.step()  # loss가 감소하는 방향으로 가중치를 업데이트
            # loss_arr[i].append(loss.cpu().detach().numpy())
            loss_arr[i][j] = loss.item()

            if j == len(x_train_loader) - 1:  # 하나의 epoch를 보면 아래를 실행.
                print("Epoch :", i + 1, " Loss :", sum(loss_arr[i], 0.0) / len(loss_arr[i]))  # 평균 loss 출력

                train_acc.append(eval(x_train_loader, y_train_loader, model))
                print("Accuracy of Train Data : {}".format(train_acc[i]))

                val_acc.append(eval(x_val_loader, y_val_loader, model))
                print("Accuracy of Validation Data : {}".format(val_acc[i]))
                loss_arr.append(loss.cpu().detach().numpy())

        print("running time :", time.time() - start)
        print('---------------------------------------------------------')

        # learning rate decay
        lr = learning_rate * (0.1 ** (i // learing_rate_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #print(param_group['lr'])

    f1_score(x_val_loader, y_val_loader, model, device)

    # Test Data에 대한 Accuracy
    print("Accuracy of Test Data : {}".format(eval(x_test_loader, y_test_loader, model)))

    # 모델 저장
    torch.save(model.state_dict(), './test1.pth')

if __name__ == '__main__':
    main()

    # validation set에 대한 precision, recall, f1-score
