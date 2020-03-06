import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AWing(nn.Module):

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)

    def forward(self, y_pred , y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = torch.abs(y-y_pred) < self.theta
        case2_ind = torch.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
        return lossMat

class Loss_weighted(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)

    def forward(self, y_pred , y, M):
        M = M.float()
        Loss = self.Awing(y_pred,y)
        weighted = Loss * (self.W * M + 1.)
        return weighted.mean()

if __name__=="__main__":

    out    = torch.randn(4,69,64,64)
    target = torch.randn(4,69,64,64)
    M = torch.randn(4,69,64,64)
    criterion = Loss_weighted()
    lossV = criterion(out,target,M)
    print(lossV)
    #loss vis
    """
    import matplotlib.pyplot as plt
    lossmap = lossV.detach()[0,-1].numpy()
    plt.imshow(lossmap)
    plt.show()
    """
