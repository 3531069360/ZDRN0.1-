import torch
import torch.nn as nn

# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super(AttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_length, in_channels = x.size()
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_length, in_channels)
        if output.size(1) == 1:
            output = output.squeeze(1)
        return output


# 定义并行分支模块
class ParallelBranch(nn.Module):
    def __init__(self, input_size, hidden_size,
                 reg_type='l2', alpha=0.5, l1_ratio=0.5,
                 connection_density=0.7):
        super(ParallelBranch, self).__init__()
        self.scattered_layer = nn.Linear(input_size, hidden_size)
        self.attention = AttentionModule(hidden_size)
        self.activation = nn.SiLU()
        self.shortcut = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = self.scattered_layer(x)
        out = self.attention(out)
        return self.activation(out + identity)


# 定义 ZDRN 模型
class ZDRN(nn.Module):
    def __init__(self, input_size, num_branches, num_layers, output_size,
                 reg_types=('l2', 'l1'), alpha=0.5, l1_ratio=0.5, connection_density=0.7):
        super(ZDRN, self).__init__()
        self.layers = nn.ModuleList()
        in_size = input_size
        for _ in range(num_layers):
            layer_branches = nn.ModuleList([
                ParallelBranch(
                    input_size=in_size,
                    hidden_size=64,
                    reg_type=reg_types[i % len(reg_types)],
                    alpha=alpha * (i + 1) / num_branches,
                    l1_ratio=l1_ratio,
                    connection_density=connection_density
                )
                for i in range(num_branches)
            ])
            self.layers.append(layer_branches)
            in_size = 64 * num_branches
        self.fc = nn.Linear(in_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            branch_outputs = []
            for branch in layer:
                branch_outputs.append(branch(x))
            x = torch.cat(branch_outputs, dim=1)
        return self.fc(x)

