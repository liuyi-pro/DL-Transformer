""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 将输入顺序(B, H, W, C)调换为[B, H//7, 7, W//7, 7, C]
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 将输入顺序中32index交换调换[B, H//7, 7, W//7, 7, C]变为[B, H//7, W//7, 7, 7, C]也就是[B, H//Mh, W//Mw, Mh, Mw, C]
    # 注意这里都是可以被window size整除的 已经处理过了 这里只是切分为window size*window size
    # contiguous()让permute之后内存不连续的变为内存连续
    # view(-1, window_size, window_size, C) -1自动推理维度, H//windowSize, W//windowSize就是窗口个数num_windows
    # 也就变为了[B*num_windows, Mh, Mw, C] Mh Mw分别为window高 宽
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image 划分window之前的H
        W (int): Width of image  划分window之前的W

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # windows.shape[0]=num_windows*B, H * W / window_size / window_size=num_windows

    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C] 输入维度分解
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # 和划分一样的 反过来罢了 恢复[B, H, W, C]
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding  包含图片切分和线性embedding两个部分的内容
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):  # 下采样倍数（每个图像块的size）；输入通道数量；输出通道数C；layernorm
        super().__init__()
        patch_size = (patch_size, patch_size)  # 4*4
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 使用卷积层来做
        # 输入通道为RGB的3；输出通道为C；卷积核kernel为4*4；步长为4；
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # 如果layerNorm有指定 那就LayerNorm 没有就线性映射

    def forward(self, x):
        _, _, H, W = x.shape  # 获取输入图片的尺寸

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:  # 0/1
            # to pad the last 3 dimensions,  PatchSize为4 起码padding3pixel
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],  # 右边padding
                          0, self.patch_size[0] - H % self.patch_size[0],  # 下面padding
                          0, 0))
        # the last 3 dimensions, use(padding_left,padding_right,padding_top,padding_bottom,padding_front, padding_back).
        # pad方法可以pad最后三个维度的内容，从后向前依次pad
        # 此时传入的x的组织形式是：x = [B, C, H, W]，只能pad后三个维度，即 C H W维度
        # 从后向前pad 即先pad W, 再pad H, 再pad C
        # W pad在右侧, H pad在下方, C不pad

        # 下采样patch_size倍
        x = self.proj(x)  # 下采样
        _, _, H, W = x.shape  # 下采样后高宽
        # flatten: [B, C, H, W] -> [B, C, HW]  batch，channel，HW 从0 1 2开始展平，从H展平，将HW展平
        # transpose: [B, C, HW] -> [B, HW, C]  然后将展平后1 2元素交换位置
        x = x.flatten(2).transpose(1, 2)  # 从0 1 2开始展平，从H展平，将HW展平。然后将展平后1 2元素交换位置
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:                                                                                # patch merging以2*2pixel理解 而不是2*2patch
        dim (int): Number of input channels.                                             # 输入channel数量
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm    # 使用何种layerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 论文中 merging层将本来要输出4C维度降低为2C维度 该操作由线性层linear实现
        self.norm = norm_layer(4 * dim)                           #

    def forward(self, x, H, W):  #
        """
        x: B, H*W, C；  x的是以batch，h*w，channel数量的形式存储；因为H*W不能知道H和W分别是多少，所以传参有H和W
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"  # 如果H * W不等于传入的参数H W的乘积，报错

        x = x.view(B, H, W, C)  # 将H W分开

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            # pad方法可以pad最后三个维度的内容，从后向前依次pad
            # 此时传入的x的组织形式是：view过了 x变了 x = [B, H, W, C]，只能pad后三个维度，即 H, W, C维度
            # 从后向前pad 即先pad C, 再pad W, 再pad H
            # C不pad W pad在右侧 H pad在下方

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C] 以步长为2进行采样 起始点0 0
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C] 以步长为2进行采样 起始点1 0
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] 以步长为2进行采样 起始点0 1
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C] 以步长为2进行采样 起始点1 1
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]  以最后一维度进行拼接 即channel变深 B H/2 W/2 4C
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]  # 整理一下 变回 HW乘一起的 HW下采样两倍 所以HW/4

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]  # 通过linear层 将4C缩减为2C

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):  # 实现窗口多头自注意力机制W MSA 因为SW MSA的移位已经在Block中做完 后续的操作实质都是W MSA
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # C 2C 4c 8c
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 每个头的dim a1乘WQ WK WV得的q1 k1 v1等的长度除以头数得q11..  C 2C 4C 8C//[3,6,12,24] =[96//3,24,24,24]
        self.scale = head_dim ** -0.5  # 1/根号d d为ki的维度 也是每个head分配到的kii的维度 因为QKV的linear维度为3C K维度为C C//head

        # define a parameter table of relative position bias 位置偏移矩阵
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        # 每个head位置偏移矩阵不一样

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # [0,1]
        coords_w = torch.arange(self.window_size[1])  # [0,1]
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # np.meshgrid需要使用 indexing="ij"属性，torch.meshgrid不需要这个属性
        # 实际上是因为torch版本不一致 torch1.7.1不需要 torch1.10需要
        # 将coords_h w的[0, ..., window size]的生成网格
        # torch.meshgrid会返回两个window_size[0], window_size[1]的矩阵，
        # 第一个矩阵是coords_h组成第一列的列向量并且复制window_size[1]列次 window_size[0]行 window_size[1]列
        # [[0, 0, 0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1, 1],
        # [2, 2, 2, 2, 2, 2, 2],
        # [3, 3, 3, 3, 3, 3, 3],
        # [4, 4, 4, 4, 4, 4, 4],
        # [5, 5, 5, 5, 5, 5, 5],
        # [6, 6, 6, 6, 6, 6, 6]],]
        # 第2个矩阵是coords_w组成第一行的行向量并且复制window_size[0]行次  window_size[0]行 window_size[1]列
        # [[0, 0, 0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1, 1],
        # [2, 2, 2, 2, 2, 2, 2],
        # [3, 3, 3, 3, 3, 3, 3],
        # [4, 4, 4, 4, 4, 4, 4],
        # [5, 5, 5, 5, 5, 5, 5],
        # [6, 6, 6, 6, 6, 6, 6]],] 然后stack在一起 变成 2， window_size[0]， window_size[1]
        coords_flatten = torch.flatten(coords, 1)  # 展平[2, Mh*Mw]
        # 展平第1个矩阵[0,0,1,1]每个元素对应的行标
        # 展平第2个矩阵[0,1,0,1]每个元素对应的列标 绝对索引
        #
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # [2, Mh*Mw, Mh*Mw]=[2, Mh*Mw, 1] - [2, 1, Mh*Mw] 需要广播broadcast 将1的维度复制Mh*Mw次
        # 相当于[2, Mh*Mw, Mh*Mw] - [2, Mh*Mw, Mh*Mw]这样做是为了所有的元素绝对索引互相减去 获得二元相对索引
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0 二元索引变一维索引 行标+2M-1
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 列标+2M-1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 行标*（2M-1）
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw] 行列求和 获得一元相对索引
        self.register_buffer("relative_position_index", relative_position_index)  # 不变了 放入模型缓存中

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 获得QKV
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # dim = total_embed_dim = 96
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)  # relative_position_bias_table初始化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # [batch_size*num_windows, Mh*Mw window内元素个数, total_embed_dim]
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head] 为了好把qkv分开
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale  # Q/根号d
        attn = (q @ k.transpose(-2, -1))  # Q * K转置

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # 根据相对索引去relative_position_bias_table中取position_bias
        # 先展平是因为relative_position_bias_table是平的 取完再转回去
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)  # Q * K转置 /根号d + B
        # [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw] + [num_heads, Mh*Mw, Mh*Mw].unsqueeze(0)
        # 给每个头依次加上pos bias, 2*2变成4*4的意义就是是对于每个头的相对位置索引 每个头的相对位置量级也不一样
        # 这样就获取了每个头的顺序 和每个头内每个元素的顺序

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]  nW=num window
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]  total_embed_dim = C
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 将每个head的QKV拼接total_embed_dim
        x = self.proj(x)  # 线性层 这个是MSA的线性层 不是最后的分类线性层 这个目的只是让QKV融合 Wo映射
        x = self.proj_drop(x)  # [nW*B, Mh*Mw, C]输出 linear前后维度不变的
        # linear处理三维矩阵 只改变最后一维维数
        # @矩阵乘法只乘最后两维度
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.  # 构建每一个block; stage比block大,stage含几个block

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, pad_size_modify=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  #
        self.shift_size = shift_size
        self.pad_size_modify = pad_size_modify  # 新加入的变量
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # shift_size如果大于windowSize会报错
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(  # shift_size=0 WMSA；否则 SWMSA
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # mlp放大倍数4,放大channel
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # MLP类 act_layer GELU激活函数

    def cal_square_modify(self, x):
        # 本来打算按batch循环 但是直接对矩阵操作最为简单
        # batch = x.shape[0]

        list_pad_mode = []
        H_2_int = x.shape[1] // 2  # x 的形式是 B, H, W, C
        W_2_int = x.shape[2] // 2

        # B, 1
        x_temp_0 = torch.norm(x[:, :H_2_int, :W_2_int, :].flatten(1).float(), p='fro', dim=1, keepdim=False, out=None, dtype=None)
        x_temp_1 = torch.norm(x[:, :H_2_int, W_2_int:, :].flatten(1).float(), p='fro', dim=1, keepdim=False, out=None, dtype=None)
        x_temp_2 = torch.norm(x[:, H_2_int:, :W_2_int, :].flatten(1).float(), p='fro', dim=1, keepdim=False, out=None, dtype=None)
        x_temp_3 = torch.norm(x[:, H_2_int:, W_2_int:, :].flatten(1).float(), p='fro', dim=1, keepdim=False, out=None, dtype=None)
        # print(x_temp_0, x_temp_1, x_temp_2, x_temp_3)  # [128,1] concate [128,4] max的index变为mode
        # print(x_temp_0.shape)

        x_temp = torch.cat([x_temp_0.unsqueeze(1), x_temp_1.unsqueeze(1), x_temp_2.unsqueeze(1), x_temp_3.unsqueeze(1)],
                           dim=-1)  # 在norm维度拼接 具体还需要gra
        # print(x_temp.shape)
        x_max, mode = torch.max(x_temp, dim=1)  # mode N 就是N区域的L2范数更大
        # print(x_max, mode)
        return mode

    def cut_square_modify(self, x, mode, pad_size_modify):
        # 本来想的 对矩阵先切割 再pad0 但是不如直接用0替换原来的元素
        # 但是那样不行 mode=0 应该是先删去右下角对应的区域 然后再在左上角pad0
        # x的形式是B, H, W, C
        cut_x = torch.zeros([x.shape[0], x.shape[1] - pad_size_modify, x.shape[2] - pad_size_modify, x.shape[3]], device=x.device)
        # print('cut_x', cut_x.shape)
        for i in range(x.shape[0]):  # batch
            if mode[i] == 0:
                cut_x[i] = x[i, :-pad_size_modify, :-pad_size_modify, :]  # mode 0
            if mode[i] == 1:
                cut_x[i] = x[i, :-pad_size_modify, pad_size_modify:, :]  # mode 1
            if mode[i] == 2:
                cut_x[i] = x[i, pad_size_modify:, :-pad_size_modify, :]  # mode 2
            if mode[i] == 3:
                cut_x[i] = x[i, pad_size_modify:, pad_size_modify:, :]  # mode 3

        return cut_x

    def pad_square_modify(self, cut_x, mode, pad_size_modify):
        # 按照swinT的方式pad
        pad_x = torch.zeros(
            [cut_x.shape[0], cut_x.shape[1] + pad_size_modify, cut_x.shape[2] + pad_size_modify, cut_x.shape[3]], device=cut_x.device)
        # print('padx0', pad_x.shape)
        for i in range(cut_x.shape[0]):
            pad_l = pad_size_modify if mode[i] % 2 == 0 else 0  # mode为偶数pad左 否则pad右
            pad_r = pad_size_modify if mode[i] % 2 == 1 else 0
            pad_t = pad_size_modify if mode[i] <= 1 else 0  # mode大于1 pad下 否则pad上
            pad_b = pad_size_modify if mode[i] >= 2 else 0
            # print(pad_l, pad_r, pad_t, pad_b)
            # the last 3 dimensions, use(padding_left,padding_right,padding_top,padding_bottom,padding_front, padding_back).
            # pad方法可以pad最后三个维度的内容，从后向前依次pad
            # 此时传入的x的组织形式是：x = [B, H, W, C]，只能pad后三个维度，即 H, W, C维度
            # 从后向前pad 即先pad C, 再pad W, 再pad H
            # C不pad, pad W 左右, pad H上下
            pad_x[i] = F.pad(cut_x[i], (0, 0, pad_l, pad_r, pad_t, pad_b))  # made 不如if
        return pad_x

    def forward(self, x, attn_mask):
        H, W = self.H, self.W  # x是B, L, C 并不知道H和W BasicLayer中494给的
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x  # short cut1
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.pad_size_modify > 0:  # 需要padding
            mode = self.cal_square_modify(x)  # 计算上一层的输出的四个部分的激活norm 得到padding方式
            # 0为pad左上角 1为pad右上角 2为pad左下角 3为pad右下角

            # 切割 以0为例 切割右下角部分的self.pad_size_modify
            cut_x = self.cut_square_modify(x, mode, self.pad_size_modify)

            # pad 以0为例 pad左上角部分的self.pad_size_modify
            pad_x = self.pad_square_modify(cut_x, mode, self.pad_size_modify)
        else:
            pad_x = x
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍 在swinT原文中 因为feature map一直是window size整数倍 所以也就不需要pad
        pad_l = pad_t = 0  # pad 右边和底部 所以另外两边是pad0
        pad_r = (self.window_size - W % self.window_size) % self.window_size  # 这样写少个判断W % self.window_size ！=0
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r != 0 and pad_b != 0:
            print('padding code goes wrong!')
        x = F.pad(pad_x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:  # SW MSA
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # B, H, W, C 1高2宽 从左往右 从上到下移
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # 对移位后的shifted_x进行划分窗口[nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # view [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # view回[nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # 切分窗口拼回[B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:  # SW MSA
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # cyclic shift从左往右 从上到下移位后的移动回去
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()  # 因为右边 下边pad 所以取前H 前W的数据 这个是不足window size整数倍的pad 和roll无关

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.   # 实现每个stage模块

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size, stage_index,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        # self.shift_size = window_size // 2  # window_size=7 是在feature map上的size 第一个feature map size是56*56
        self.shift_size = 0  # window_size=7 是在feature map上的size 第一个feature map size是56*56
        self.pad_size_modify = window_size // 2  # padding size
        self.stage_index = stage_index  # modify
        # print('self.stage_index', self.stage_index)
        # for j in range(depth):
        #     pad_size_modify = self.pad_size_modify if ((j == 1) and (self.stage_index < 2)) or (j == 5) else 0
        #     print(pad_size_modify)

        # build blocks 创建block组成stage
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                # shift_size=0 if (i % 2 == 0) else self.shift_size,  # 每个stage内W-MSA和SW-MSA接替使用 先W-MSA再SW-MSA
                shift_size=0,  # modify每个stage都是W-MSA
                # pad_size_modify=0 if (i % 2 == 0) else self.pad_size_modify,  # depth=[2,2,6,2]
                pad_size_modify=self.pad_size_modify if ((i == 1) and (self.stage_index < 2)) or (i == 5) else 0,  # depth=[2,2,6,2]
                mlp_ratio=mlp_ratio,  # mlpBLock和vit一样 第一个全连接层将维度翻4倍
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])  # depth是依次传入的该stage堆叠block的次数 [2, 2, 6, 2]

        # patch merging layer 在每个stage的末尾 和论文图换个说法 其实是一样的
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size  # 先向上取整 再乘回来 相当于放大到整数倍（padding后的size）
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),  # -7 完整不被分割的
                    slice(-self.window_size, -self.shift_size),  # -7 -3 被shift一部分后剩下的部分
                    slice(-self.shift_size, None))  # -3  shift掉的部分
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # 形成在高度和宽度方向上 从左到右的 从上到下递增的mask 类似于
        '''
        0 0 1 2
        0 0 1 2
        0 0 1 2 
        3 3 4 5
        6 6 7 8'''

        mask_windows = window_partition(img_mask, self.window_size)
        # [nW, Mh, Mw, 1] [窗口数, windowSize, windowSize, 1] 传入的是img_mask channel为1 batch为1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [窗口数, Mh*Mw]  [窗口数行，窗口内元素列]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 加一个维度
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]广播机制 nw依次动, 1行Mh*Mw列 将这1行复制Mh*Mw次；Mh*Mw行1列 将这1列复制Mh*Mw次
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # attn_mask==0就是mask_windows的index相同的值，说明该pixel与对角线上元素是相邻的 否则给attn_mask=-100
        # 具体怎么看attn_mask呢？
        # 每行attn_mask为0的pixel都与该行对角线上元素为同区域 可做attention
        #
        # 输出[nW, Mh*Mw, Mh*Mw] 也就是窗口数个attn_mask 该窗口可以和哪些部分做attention一目了然
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:  # self.blocks为当前stage的全部堆叠
            blk.H, blk.W = H, W  # 给当前blk添加高宽属性
            if not torch.jit.is_scripting() and self.use_checkpoint:  # 不使用 不进入
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # 每个blk都是SwinTransformerBlock
        if self.downsample is not None:  # patch merging层
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2  # 下采样两倍 并且返回 +1是为了防止奇数

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4                                   下采样倍数
        in_chans (int): Number of input image channels. Default: 3                              输入通道数
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96                                 论文中的C 线性embedding层输出维度
        depths (tuple(int)): Depth of each Swin Transformer layer.                              每个stage内swinT Block重复次数
        num_heads (tuple(int)): Number of attention heads in different layers.                  multi head self-attention头个数
        window_size (int): Window size. Default: 7                                              WMSA和SWMSA的窗口大小
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4                 mlpBLock和vit一样 第一个全连接层将通道维度翻4倍
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True      MSA模块是不是使用偏执
        drop_rate (float): Dropout rate. Default: 0                                             在Patch embed层后用的，pos_drop和mlp中用
        attn_drop_rate (float): Attention dropout rate. Default: 0                              在MSA过程中用的drop rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.1                             在swinT Block中用的drop rate
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.                     默认使用LayerNorm
        patch_norm (bool): If True, add normalization after patch embedding. Default: True      在Patch embed层后
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)  # 4个stage内的Block堆叠 4个stage
        self.embed_dim = embed_dim     # C
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # stage4后输出的通道数 即8C 2^(4-1)=8
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches 将图片划分为不重叠的patch并且映射为C通道
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)  # dropout层接在patch_embed层后

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # 各个swinT Block的dropout rate是从浅层到深度逐渐从0增加到drop_path_rate
        # 所以用了linspace (2, 2, 6, 2)每个层的dropout rate都不一样

        # build layers 遍历生成每个层对应的配置
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # num_layers=4，4个stage
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            # 相当于论文中的虚线框 向右移动了一下

            # 需要pad位置修改 pad%3过于粗暴 在这里改modify

            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),  # C 2C 4c 8c
                                depth=depths[i_layer],  # (2, 2, 6, 2) 每个stage依次传入该stage内的block的堆叠次数
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                stage_index=i_layer,  # modifyV3.1
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,  # 前3组layers有patch merging和论文图不太一样 但是是一个东西
                                use_checkpoint=use_checkpoint,
                                )  # 没用上的api
            self.layers.append(layers)  # 每次创建一个block再append 最终组成一个stage

        self.norm = norm_layer(self.num_features)  # stage4后输出的通道数 即8C=8*96
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应全局平均池化
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类头
        # 输入stage4后输出的通道数 即8C=8*96
        # 输出维度为总类别数

        self.apply(self._init_weights)  # 初始化参数

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C] ,L=H*W
        x, H, W = self.patch_embed(x)  # patch embedding图像 下采样4倍 切分为4*4的patch
        x = self.pos_drop(x)  # dropout层 引入随机性

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)                        # x此时为 [B, L, C]
        x = self.avgpool(x.transpose(1, 2))     # x此时为 [B, C, 1]
        x = torch.flatten(x, 1)                 # 以通道展开 # x此时为 [B, C]
        x = self.head(x)                        # 输入stage4后输出的通道数 即8C=8*96 输出维度为总类别数
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


# 调试用
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = swin_tiny_patch4_window7_224(num_classes=15).to(device)
