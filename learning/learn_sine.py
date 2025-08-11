import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# learn a function f(x) that maps points x to sin(x), i.e., learn a function
# approximation of the sine function

# set seed for reproducibility
torch.manual_seed(0)

w = 4 # frequency

# generate training data: y = sin(x)
x_train = torch.linspace(-2 * torch.pi, 2 * torch.pi, 1000).unsqueeze(1)
y_train = torch.sin(w*x_train)

# Define a simple feedforward neural network
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# initialize model, loss, and optimizer
model = SineNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
n_epochs = 10000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"epoch [{epoch+1:6d} | {n_epochs:6d} ], loss: {loss.item():4.6e}")

# Plot the result
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-2 * torch.pi, 2 * torch.pi, 1000).unsqueeze(1)
    y_pred = model(x_test)

plt.plot(x_train.numpy(), y_train.numpy(), label="true f(x) = sin(x)")
plt.plot(x_test.numpy(), y_pred.numpy(), label="nn prediction f_theta(x)")
plt.legend()
plt.title("learning sin(x) with pytorch")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.savefig('sine-nn-prediction.pdf')

#plt.show()



###########################################################
# This code is part of the python toolbox termed
#
# CHAMELEON --- Computational and mAthematical MEthods in
# machine LEarning, Optimization and iNference
#
# For details see https://github.com/andreasmang/chameleon
###########################################################

