import torch.nn as nn

class Text_CNN(nn.Module):
    def __init__(self):
        super(Text_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=127, out_channels=512, kernel_size=6, stride=1),
            # [batch_size, 512, 195]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
            # [batch_size, 512, 193]
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=9, stride=1),
            # [batch_size, 512, 185]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
            # [batch_size, 512, 183]
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=12, stride=1),
            # [batch_size, 512, 172]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
            # [batch_size, 512, 170]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * 170, out_features=1024),
            # [batch_size, 86016] -> [batch_size, 1024]
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            # [batch_size, 1024] -> [batch_size, 1024]
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Linear(in_features=1024, out_features=1)
        # [batch_size, 1024] -> [batch_size, 2]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # [batch_size, 512, 120]
        x = self.conv2(x)
        # [batch_size, 512, 110]
        x = self.conv3(x)
        # [batch_size, 512, 97]

        # collapse
        x = x.view(x.size(0), -1)  # [batch_size, 49664]
        # linear layer
        x = self.fc1(x)  # [batch_size, 1024]
        # linear layer
        x = self.fc2(x)  # [batch_size, 1024]
        # linear layer
        x = self.fc3(x)  # [batch_size, 2]
        # output layer
        x = self.sigmoid(x)

        return x