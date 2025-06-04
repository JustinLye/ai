import torch
import torch.nn
import torch.nn.functional




class network(torch.nn.Module):
  def __init__(self, action_size, seed=42) -> None:
      super(network, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.convolution_layer0 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
      self.batch_normalization0 = torch.nn.BatchNorm2d(32)
      self.convolution_layer1 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
      self.batch_normalization1 = torch.nn.BatchNorm2d(64)
      self.convolution_layer2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
      self.batch_normalization2 = torch.nn.BatchNorm2d(64)
      self.convolution_layer3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1)
      self.batch_normalization3 = torch.nn.BatchNorm2d(128)

      self.full_connection0 = torch.nn.Linear(10 * 10 * 128, 512)
      self.full_connection1 = torch.nn.Linear(512, 256)
      self.full_connection2 = torch.nn.Linear(256, action_size)

  def forward(self, state):
    signal = torch.nn.functional.relu(self.batch_normalization0(self.convolution_layer0(state)))
    signal = torch.nn.functional.relu(self.batch_normalization1(self.convolution_layer1(signal)))
    signal = torch.nn.functional.relu(self.batch_normalization2(self.convolution_layer2(signal)))
    signal = torch.nn.functional.relu(self.batch_normalization3(self.convolution_layer3(signal)))
    signal = signal.view(signal.size(0), -1) # flatten
    signal = torch.nn.functional.relu(self.full_connection0(signal))
    signal = torch.nn.functional.relu(self.full_connection1(signal))
    return self.full_connection2(signal)
