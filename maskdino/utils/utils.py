# import torch
# import copy
# from torch import nn, Tensor
# import os

# import math
# import torch.nn.functional as F
# from torch import nn


# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x


# def inverse_sigmoid(x, eps=1e-5):
#     x = x.clamp(min=0, max=1)
#     x1 = x.clamp(min=eps)
#     x2 = (1 - x).clamp(min=eps)
#     return torch.log(x1/x2)


# def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
#     """
#     Input:
#         - memory: bs, \sum{hw}, d_model
#         - memory_padding_mask: bs, \sum{hw}
#         - spatial_shapes: nlevel, 2
#     Output:
#         - output_memory: bs, \sum{hw}, d_model
#         - output_proposals: bs, \sum{hw}, 4
#     """
#     N_, S_, C_ = memory.shape
#     base_scale = 4.0
#     proposals = []
#     _cur = 0
#     for lvl, (H_, W_) in enumerate(spatial_shapes):
#         mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
#         valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
#         valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

#         grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
#                                         torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
#         grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

#         scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
#         grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
#         wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
#         proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
#         proposals.append(proposal)
#         _cur += (H_ * W_)
#     output_proposals = torch.cat(proposals, 1)
#     output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
#     output_proposals = torch.log(output_proposals / (1 - output_proposals))
#     output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
#     output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

#     output_memory = memory
#     output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
#     output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
#     return output_memory, output_proposals


# def gen_sineembed_for_position(pos_tensor):
#     # n_query, bs, _ = pos_tensor.size()
#     # sineembed_tensor = torch.zeros(n_query, bs, 256)
#     scale = 2 * math.pi
#     dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
#     dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / 128)
#     x_embed = pos_tensor[:, :, 0] * scale
#     y_embed = pos_tensor[:, :, 1] * scale
#     pos_x = x_embed[:, :, None] / dim_t
#     pos_y = y_embed[:, :, None] / dim_t
#     pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
#     if pos_tensor.size(-1) == 2:
#         pos = torch.cat((pos_y, pos_x), dim=2)
#     elif pos_tensor.size(-1) == 4:
#         w_embed = pos_tensor[:, :, 2] * scale
#         pos_w = w_embed[:, :, None] / dim_t
#         pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

#         h_embed = pos_tensor[:, :, 3] * scale
#         pos_h = h_embed[:, :, None] / dim_t
#         pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

#         pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
#     else:
#         raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
#     return pos


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     if activation == "prelu":
#         return nn.PReLU()
#     if activation == "selu":
#         return F.selu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# def _get_clones(module, N, layer_share=False):

#     if layer_share:
#         return nn.ModuleList([module for i in range(N)])
#     else:
#         return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# import torch
# import copy
# from torch import nn, Tensor
# import os

# import math
# import torch.nn.functional as F
# from torch import nn


# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x


# def inverse_sigmoid(x, eps=1e-5):
#     x = x.clamp(min=0, max=1)
#     x1 = x.clamp(min=eps)
#     x2 = (1 - x).clamp(min=eps)
#     return torch.log(x1/x2)


# def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
#     """
#     Input:
#         - memory: bs, \sum{hw}, d_model
#         - memory_padding_mask: bs, \sum{hw}
#         - spatial_shapes: nlevel, 2
#     Output:
#         - output_memory: bs, \sum{hw}, d_model
#         - output_proposals: bs, \sum{hw}, 4
#     """
#     N_, S_, C_ = memory.shape
#     base_scale = 4.0
#     proposals = []
#     _cur = 0
#     for lvl, (H_, W_) in enumerate(spatial_shapes):
#         mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
#         valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
#         valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

#         grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
#                                         torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
#         grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

#         scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
#         grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
#         wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
#         proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
#         proposals.append(proposal)
#         _cur += (H_ * W_)
#     output_proposals = torch.cat(proposals, 1)
#     output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
#     output_proposals = torch.log(output_proposals / (1 - output_proposals))
#     output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
#     output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

#     output_memory = memory
#     output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
#     output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
#     return output_memory, output_proposals


# def gen_sineembed_for_position(pos_tensor):
#     # n_query, bs, _ = pos_tensor.size()
#     # sineembed_tensor = torch.zeros(n_query, bs, 256)
#     scale = 2 * math.pi
#     dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
#     dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / 128)
#     x_embed = pos_tensor[:, :, 0] * scale
#     y_embed = pos_tensor[:, :, 1] * scale
#     pos_x = x_embed[:, :, None] / dim_t
#     pos_y = y_embed[:, :, None] / dim_t
#     pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
#     if pos_tensor.size(-1) == 2:
#         pos = torch.cat((pos_y, pos_x), dim=2)
#     elif pos_tensor.size(-1) == 4:
#         w_embed = pos_tensor[:, :, 2] * scale
#         pos_w = w_embed[:, :, None] / dim_t
#         pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

#         h_embed = pos_tensor[:, :, 3] * scale
#         pos_h = h_embed[:, :, None] / dim_t
#         pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

#         pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
#     else:
#         raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
#     return pos


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     if activation == "prelu":
#         return nn.PReLU()
#     if activation == "selu":
#         return F.selu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# def _get_clones(module, N, layer_share=False):

#     if layer_share:
#         return nn.ModuleList([module for i in range(N)])
#     else:
#         return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


import torch
import copy
from torch import nn, Tensor
import os

import math
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals


#############################
class QueryProposal(nn.Module):

    def __init__(self, num_features, num_queries, num_classes):
        super().__init__()
        self.topk = num_queries
        self.num_classes = num_classes

        self.conv_proposal_cls_logits = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.dp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(0, 1, h, device=x.device)
        x_loc = torch.linspace(0, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        locations = torch.stack([x_loc, y_loc], 0).unsqueeze(0)
        return locations

    def seek_local_maximum(self, x, epsilon=1e-6):
        """
        inputs:
            x: torch.tensor, shape [b, c, h, w]
        return:
            torch.tensor, shape [b, c, h, w]
        """
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        # top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
        maximum = (x >= x_pad[:, :, :-2, 1:-1]) & \
                  (x >= x_pad[:, :, 2:, 1:-1]) & \
                  (x >= x_pad[:, :, 1:-1, :-2]) & \
                  (x >= x_pad[:, :, 1:-1, 2:]) & \
                  (x >= x_pad[:, :, :-2, :-2]) & \
                  (x >= x_pad[:, :, :-2, 2:]) & \
                  (x >= x_pad[:, :, 2:, :-2]) & \
                  (x >= x_pad[:, :, 2:, 2:]) & \
                  (x >= epsilon)
        return maximum.to(x)

    def forward(self, inp):
        # b, c, _, _ = x.shape
        # topk_proposals = self.conv(x.reshape(b, c, -1)).permute(0, 2, 1)
        # x = inp[1] + self.dp(inp[2])

        x = inp[1] + self.up(inp[0]) + self.dp(inp[2])
        # x = inp[1]
        proposal_cls_logits = self.conv_proposal_cls_logits(x)  # b, c, h, w
        proposal_cls_probs = proposal_cls_logits.softmax(dim=1)  # b, c, h, w
        # proposal_cls_probs = proposal_cls_probs.mul(proposal_cls_one_hot)
        # proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)  # b, c, h, w
        # proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map  # b, c, h, w

        #################XQ
        # import cv2
        # import numpy as np
        # import matplotlib.pyplot as plt

        # show = F.interpolate(proposal_cls_probs, size=(384, 384), mode='bilinear', align_corners=False)
        # topk_indices_show = torch.topk(proposal_cls_probs[:, :, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1][0]
        # # topk_indices_show = torch.topk(proposal_cls_probs[:, :-1, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1][0]  # b, q

        # y_indices = topk_indices_show // 24
        # x_indices = topk_indices_show % 24

        # y_indices = y_indices.cpu().numpy()
        # x_indices = x_indices.cpu().numpy()

        # y_indices_ratio = y_indices / 24
        # x_indices_ratio = x_indices / 24

        # y_indices = y_indices_ratio * 384
        # x_indices = x_indices_ratio * 384

        # mark_color = (0, 0, 255)  # (B, G, R) - red
        # white_image = np.ones((384, 384, 3), dtype=np.uint8) * 255

        # for y_indice, x_indice in zip(y_indices, x_indices):
        #     cv2.circle(white_image, (int(x_indice), int(y_indice)), radius=3, color=mark_color, thickness=-1)
        
        # white_image = np.transpose(white_image, (2, 0, 1))
        # cv2.imwrite('image_with_markers.png', cv2.cvtColor(np.transpose(white_image, (1, 2, 0)), cv2.COLOR_RGB2BGR))
        #################

        # num_g = 0
        # num_f = 0
        # fg_index = proposal_cls_probs[:, :, :, :].flatten(2).max(1)[1][0]
        # # top-k indices
        topk_indices = torch.topk(proposal_cls_probs[:, :, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1]
        # fg_indices = torch.topk(proposal_cls_probs[:, :, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1][0]  # b, q
        # for i in fg_indices:
        #     fg = fg_index[i]
        #     if fg == 0:
        #         num_g += 1
        #     elif fg == 1:
        #         num_f += 1

        # # 将数量写入txt文件
        # with open('count_result.txt', 'a') as f:
        #     f.write(f"Foreground: {num_f}")
        #     f.write(f"Background: {num_g}\n")


        topk_indices = topk_indices.unsqueeze(1)  # b, 1, q

        # topk queries
        topk_proposals = torch.gather(x.flatten(2), dim=2, index=topk_indices.repeat(1, x.shape[1], 1))  # b, c, q
        # pos_embeddings = pos_embeddings.repeat(x.shape[0], 1, 1, 1).flatten(2)
        # topk_pos_embeddings = torch.gather(
        #     pos_embeddings, dim=2, index=topk_indices.repeat(1, pos_embeddings.shape[1], 1)
        # )  # b, c, q
        # if self.training:
        #     locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
        #     topk_locations = torch.gather(
        #         locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
        #     )
        #     topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
        # else:
        #     topk_locations = None
        return topk_proposals, proposal_cls_logits
#############################


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N, layer_share=False):

    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
