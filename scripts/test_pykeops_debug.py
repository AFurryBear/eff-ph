# Create two arrays with 3 columns and a (huge) number of lines, on the GPU
import torch  # NumPy, Matlab and R are also supported
M, N, D = 1000000, 2000000, 3
x = torch.randn(M, D, requires_grad=True).cuda()  # x.shape = (1e6, 3)
y = torch.randn(N, D).cuda()                      # y.shape = (2e6, 3)

# Turn our dense Tensors into KeOps symbolic variables with "virtual"
# dimensions at positions 0 and 1 (for "i" and "j" indices):
from pykeops.torch import LazyTensor
x_i = LazyTensor(x.view(M, 1, D))  # x_i.shape = (1e6, 1, 3)
y_j = LazyTensor(y.view(1, N, D))  # y_j.shape = ( 1, 2e6,3)

# We can now perform large-scale computations, without memory overflows:
D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (1e6,2e6,1) matrix of squared distances
K_ij = (- D_ij).exp()               # Symbolic (1e6,2e6,1) Gaussian kernel matrix

# We come back to vanilla PyTorch Tensors or NumPy arrays using
# reduction operations such as .sum(), .logsumexp() or .argmin()
# on one of the two "symbolic" dimensions 0 and 1.
# Here, the kernel density estimation   a_i = sum_j exp(-|x_i-y_j|^2)
# is computed using a CUDA scheme that has a linear memory footprint and
# outperforms standard PyTorch implementations by two orders of magnitude.
a_i = K_ij.sum(dim=1)  # Genuine torch.cuda.FloatTensor, a_i.shape = (1e6, 1), 

# Crucially, KeOps fully supports automatic differentiation!
g_x = torch.autograd.grad((a_i ** 2).sum(), [x])