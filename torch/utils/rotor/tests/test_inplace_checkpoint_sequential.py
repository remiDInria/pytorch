import torch
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn as nn

# create a simple Sequential model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU()
)

# create the model inputs
input_var = Variable(torch.randn(1, 100), requires_grad=True)

# set the number of checkpoint segments
segments = 2

# get the modules in the model. These modules should be in the order
# the model should be executed
modules = [module for k, module in model._modules.items()]

# now call the checkpoint API and get the output
out = checkpoint_sequential(modules, segments, input_var)

# run the backwards pass on the model. For backwards pass, for simplicity purpose,
# we won't calculate the loss and rather backprop on out.sum()
model.zero_grad()
out.sum().backward()

# now we save the output and parameter gradients that we will use for comparison purposes with
# the non-checkpointed run.
output_checkpointed = out.data.clone()
grad_checkpointed = {}
for name, param in model.named_parameters():
    grad_checkpointed[name] = param.grad.data.clone()