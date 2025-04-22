import torch
import torch.nn as nn


class RaagRecog(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # keeps [n_mels, time] size
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 1)),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # collapse mel axis, keep time axis
        )
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        output, (h_n, c) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.reshape(h_n.size()[0], -1)
        return self.fc(h_n)
