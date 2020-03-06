import torch
import torch.nn as nn

class AddCoords(nn.Module):

    def __init__(self, with_r=False, with_boundary=False):
        super().__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, boundary_map):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and (boundary_map is not None):
            # B, 64(H), 64(W)
            boundary_map = boundary_map.view(boundary_map.shape[0],1,boundary_map.shape[1],boundary_map.shape[2])
            boundary_channel = torch.clamp(boundary_map,0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel>0.05,
                                              xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel>0.05,
                                              yy_channel, zero_tensor)

            ret = torch.cat([ret, xx_channel, yy_channel], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=True, with_boundary=False,**kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r,with_boundary=with_boundary)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        if with_boundary:
            in_size += 2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x, boundary_map=None):
        ret = self.addcoords(x, boundary_map)
        ret = self.conv(ret)
        return ret


if __name__=="__main__":
    x = torch.randn(1,3,633,357)
    coordconv1 = CoordConv(3,64,kernel_size=3,padding=1)
    out = coordconv1(x)
    print(out.shape)
