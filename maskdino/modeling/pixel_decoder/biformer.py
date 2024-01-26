import torch

from torch import nn, einsum, Tensor
import copy
from .position_encoding import PositionEmbeddingLearned
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional, List
import torch.nn.functional as F
# from .position_encoding import build_position_encoding

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(200, num_pos_feats)
        self.col_embed = nn.Embedding(200, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i) # *****        
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        return pos

#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         h, w = x.shape[-2:]
#         i = torch.arange(w, device=x.device)
#         j = torch.arange(h, device=x.device)
#         x_emb = self.col_embed(i)
#         y_emb = self.row_embed(j)
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
#         return pos
    
def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    
# class Transformer_self_attention(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4,
#                  add_channel_attention=False,
#                  use_deformable_box_attn=False,
#                  box_attn_type='roi_align',
#                  ):
#         super().__init__()
        
#         self.to_qkv = nn.Linear(d_model, d_model * 3, bias = False)
        
#         self.heads = n_heads
#         # self attention
#         if use_deformable_box_attn:
#             self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
#         else:
#             self.self_attn = nn.MultiheadAttention(d_model, n_heads)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation, d_model=d_ffn)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # channel attention
#         self.add_channel_attention = add_channel_attention
#         if add_channel_attention:
#             self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
#             self.norm_channel = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward(self, src):
        
#         qkv = self.to_qkv(src).chunk(3, dim = -1)
#         q, k, v = qkv        
        
#         k = k.transpose(0,1)
#         v = v.transpose(0,1)
#         q = q.transpose(0,1)
        
#         # self attention
#         src2, _ = self.self_attn(q, k, v)
#         src2 = src2.transpose(0,1)
# #         import pdb; pdb.set_trace()
        
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         # ffn
#         src = self.forward_ffn(src)

#         # channel attn
#         if self.add_channel_attention:
#             src = self.norm_channel(src + self.activ_channel(src))

#         return src

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src = src.transpose(0, 1)
#         import pdb; pdb.set_trace()
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos).transpose(0, 1)
#         print("self-attn ")
        return self.forward_post(src, src_mask, src_key_padding_mask, pos).transpose(0, 1)
    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos).transpose(0, 1)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos).transpose(0, 1)
    
# class Transformer_cross_attention(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4,
#                  add_channel_attention=False,
#                  use_deformable_box_attn=False,
#                  box_attn_type='roi_align',
#                  ):
#         super().__init__()
        
#         self.to_qkv = nn.Linear(d_model, d_model * 2, bias = False)
#         self.heads = n_heads
        
#         # self attention
#         if use_deformable_box_attn:
#             self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
#         else:
#             self.self_attn = nn.MultiheadAttention(d_model, n_heads)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation, d_model=d_ffn)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # channel attention
#         self.add_channel_attention = add_channel_attention
#         if add_channel_attention:
#             self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
#             self.norm_channel = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward(self, q, kv):
#         kv = self.to_qkv(kv).chunk(2, dim = -1)
#         k, v = kv
#         k = k.transpose(0,1)
#         v = v.transpose(0,1)
#         q = q.transpose(0,1)
        
# #         import pdb; pdb.set_trace()
        
#         # self attention
#         src2, _ = self.self_attn(q, k, v)
#         src2 = src2.transpose(0,1)
# #         import pdb; pdb.set_trace()

#         src = q + self.dropout1(src2)
#         src = self.norm1(src)

#         # ffn
#         src = self.forward_ffn(src)

#         # channel attn
#         if self.add_channel_attention:
#             src = self.norm_channel(src + self.activ_channel(src))

#         return src
    
# class Transformer_self_attention(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
    
    
# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_k = nn.Linear(dim, inner_dim , bias=False)
#         self.to_v = nn.Linear(dim, inner_dim , bias = False)
#         self.to_q = nn.Linear(dim, inner_dim, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x_q, x_kv): # ([1, 494, 256]) # ([1, 130, 256])
#         b, n, _, h = *x_kv.shape, self.heads

#         k = self.to_k(x_kv)
#         k = rearrange(k, 'b n (h d) -> b h n d', h = h)
#         v = self.to_v(x_kv)
#         v = rearrange(v, 'b n (h d) -> b h n d', h = h)
#         q = self.to_q(x_q) # self.to_q(x_q.unsqueeze(1))
#         q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         attn = dots.softmax(dim=-1)
        
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out =  self.to_out(out)
        
#         return out
    
    
    
# class Transformer_cross_attention(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 nn.LayerNorm(dim),
#                 CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x_q, x_kv):
#         for norm, attn, ff in self.layers:
#             x = attn(norm(x_q), x_kv) + norm(x_q)
#             x = ff(x) + x
#         return x
    
    
class Fake_FPN(nn.Module):
    def __init__(self, 
        embed_dim=256, 
        num_heads=8, 
        num_level_feats=4,
        mlp_dim=1024,
        dropout=0.1,
        layer_share=False, 
    ):
        super().__init__()
        self.num_level_feats = num_level_feats
        dim_head = int(embed_dim / num_heads)
#         self_attn = Transformer_self_attention(embed_dim, 1, num_heads, dim_head, mlp_dim, dropout)
#         self_attn = Transformer_self_attention(embed_dim, mlp_dim, dropout)

        self_attn = TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        self.self_attn = _get_clones(self_attn, num_level_feats, layer_share)
        
#         self.learnable_pos = PositionEmbeddingLearned(embed_dim//2)
        
#         cross_attn = Transformer_cross_attention(embed_dim, 1, num_heads, dim_head, mlp_dim, dropout)
#         cross_attn = Transformer_cross_attention(embed_dim, mlp_dim, dropout)
#         cross_attn = TransformerDecoderLayer(embed_dim, num_heads, mlp_dim, dropout)
#         self.cross_attn = _get_clones(cross_attn, num_level_feats, layer_share) 
        
    def fea_with_pos(self, fea_list):
#         import pdb; pdb.set_trace()
#         pos_list = self.learnable_pos(fea_list)        
#         fea_with_pos = fea_list + pos_list
#         bs, c, h, w = fea_with_pos.shape
#         fea_with_pos = fea_with_pos.reshape(bs, -1, c)
        
        # debug
        bs, c, h, w = fea_list.shape
        fea_with_pos = fea_list.reshape(bs, -1, c)
        
        return fea_with_pos
        
        
    def forward(self, 
            src: list, 
            ):
        """
        Input:
            - src: [[bs, 256, h, w], ...] len(src)=4
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        
#         import pdb; pdb.set_trace()
        fea_with_pos_list = []
        h_w_list = []
        for i in range(len(src)):
            fea_with_pos_list.append(self.fea_with_pos(src[i]))
            h_w_list.append(src[i].shape[:])
#         import pdb; pdb.set_trace()

        new_feas = []
        
        for i, (fea, attn_layer) in enumerate(zip(fea_with_pos_list[::-1], self.self_attn)):
            if i < 3:
                new_feas.append(attn_layer(fea))
        
        new_feas.append(fea_with_pos_list[0])
        
#         print("self-attn")
#         cross_feas = []
#         cross_feas.append(new_feas[0])
#         for i, (new_fea, cross_layer) in enumerate(zip(new_feas, self.cross_attn)):
#             if i < self.num_level_feats - 3:
#                 cross_feas.append(cross_layer(new_feas[i+1], new_fea) + new_feas[i+1])
        
        # reshape
        out_feas = []
        
#         out_feas.append(new_feas[-1].reshape(h_w_list[0]))
#         out_feas.append(new_feas[-2].reshape(h_w_list[1]))
#         out_feas.append(cross_feas[-1].reshape(h_w_list[2]))
#         out_feas.append(new_feas[-4].reshape(h_w_list[3]))
        
# #         for i, fea in enumerate(cross_feas[::-1]):
        for i, fea in enumerate(new_feas[::-1]):
#         for i, fea in enumerate(fea_with_pos_list):
            b, c, h, w = h_w_list[i]
            out_fea = fea.reshape(b, c, h, w)
            out_feas.append(out_fea)

#         import pdb; pdb.set_trace()
        
        return out_feas
    
    
class Fake_FPN_with_BiFormer(nn.Module):
    def __init__(self, 
        embed_dim=256, 
        num_heads=8, 
        num_level_feats=3,
        topk=4,
        mlp_dim=1024,
        dropout=0.1,
        layer_share=False, 
    ):
        super().__init__()
        self.num_level_feats = num_level_feats
        self.embed_dim = embed_dim
        dim_head = int(embed_dim / num_heads)

#         self_attn = TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
#         self.self_attn = _get_clones(self_attn, num_level_feats, layer_share)
        
#         self.learnable_pos = PositionEmbeddingLearned(embed_dim//2)

        cross_attn = TransformerDecoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        self.cross_attn = _get_clones(cross_attn, num_level_feats, layer_share) 
#         self.routing_act = nn.Sigmoid()
        self.topk = topk
#         self.bn = nn.BatchNorm2d(self.embed_dim)
#         self.to_k = nn.Linear(embed_dim, embed_dim , bias=False)
#         self.to_v = nn.Linear(embed_dim, embed_dim , bias = False)
        
    def fea_with_pos(self, fea_list):
#         import pdb; pdb.set_trace()
#         pos_list = self.learnable_pos(fea_list)        
#         fea_with_pos = fea_list + pos_list
#         bs, c, h, w = fea_with_pos.shape
#         fea_with_pos = fea_with_pos.reshape(bs, -1, c)
        
        # debug
        bs, c, h, w = fea_list.shape
        fea_with_pos = fea_list.reshape(bs, -1, c)
        
        return fea_with_pos
    
    def cross_with_topk(self, query, all_feas, level):
        bs, c, h, w = query.shape
        query = rearrange(query, 'bs c h w -> bs (h w) c')
        
        similarty = torch.bmm(query, all_feas.transpose(1,2)) # bs, h*w, \sum{hxw}
        topk_attn_logit, topk_index = torch.topk(similarty, k=self.topk, dim=-1) # torch.Size([2, h*w, topk])
        
        src_flatten = all_feas.reshape(-1, self.embed_dim)
        
        topk_index = topk_index.reshape(-1)
        
        # routing activation
#         r_weight = self.routing_act(topk_attn_logit) # torch.Size([2, 192, 50])
        
        topk_kv = src_flatten.index_select(0, topk_index).reshape(bs, -1, self.topk, self.embed_dim)
#         k = self.to_k(topk_kv)
#         v = self.to_v(topk_kv)
        # cross-attn
#        tmp_cross_res = []
#        for i in range(query.shape[1]):
#            for j, cross_layer in enumerate(self.cross_attn):
#                if j == level:
#                    res = cross_layer(query[:, i, :].unsqueeze(1), topk_kv[:, i, :, :])
#                    tmp_cross_res.append(res)
#        new_query = torch.cat(tmp_cross_res, 1)
        for j, cross_layer in enumerate(self.cross_attn):
            if j == level:
                new_query = cross_layer(query.reshape(-1, self.embed_dim).unsqueeze(1), topk_kv.reshape(-1, self.topk, self.embed_dim))
                new_query = new_query.reshape(bs, -1, self.embed_dim).transpose(1, 2).reshape(bs, c, h, w)
        return new_query
    
        
    def forward(self, 
            src: list, 
            ):
        """
        Input:
            - src: [[bs, 256, h, w], ...] len(src)=4
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        
#         import pdb; pdb.set_trace()
#         fea_with_pos_list = []
        
#         h_w_list = []
#         for i in range(len(src)):
# #             fea_with_pos_list.append(self.fea_with_pos(src[i]))
#             h_w_list.append(src[i].shape[:])
        
        # flatten
        src_flatten = []
        for lvl, src_ in enumerate(src):
            bs, c, h, w = src_.shape
            src_ = src_.flatten(2).transpose(1, 2) # bs, hw, c
            src_flatten.append(src_)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        
        new_feas = []
        for level, src_ in enumerate(src[::-1]):
            if level>=3:
                break
            tmp_fea = self.cross_with_topk(src_, src_flatten, level)
            new_feas.append(tmp_fea + src_)
#             new_feas.append(tmp_fea)
#         new_feas.append(src[0]) 
        
        
#         for i, (fea, attn_layer) in enumerate(zip(fea_with_pos_list[::-1], self.self_attn)):
#             if i < 3:
#                 new_feas.append(attn_layer(fea))
        
#         new_feas.append(fea_with_pos_list[0])
        
#         print("self-attn")
#         cross_feas = []
#         cross_feas.append(new_feas[0])
#         for i, (new_fea, cross_layer) in enumerate(zip(new_feas, self.cross_attn)):
#             if i < self.num_level_feats - 3:
#                 cross_feas.append(cross_layer(new_feas[i+1], new_fea) + new_feas[i+1])
        
        # reshape
#         out_feas = []
        
#         out_feas.append(new_feas[-1].reshape(h_w_list[0]))
#         out_feas.append(new_feas[-2].reshape(h_w_list[1]))
#         out_feas.append(cross_feas[-1].reshape(h_w_list[2]))
#         out_feas.append(new_feas[-4].reshape(h_w_list[3]))
        
# #         for i, fea in enumerate(cross_feas[::-1]):
#         for i, fea in enumerate(new_feas[::-1]):
#        for i, fea in enumerate(fea_with_pos_list):
#            b, c, h, w = h_w_list[i]
#            out_fea = fea.reshape(b, c, h, w)
#            out_feas.append(out_fea)

#         import pdb; pdb.set_trace()
        
        return new_feas[::-1]
