import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if r > 0:
            self.A = nn.Linear(base.in_features, r, bias=False)
            self.B = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.scaling = alpha / r
            for p in self.base.parameters():
                p.requires_grad = False
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 1.0

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            y = y + self.B(self.A(self.drop(x))) * self.scaling
        return y


class LoRAConv1d(nn.Module):
    """ LoRA for Conv1d: base(x) + B(A(x)) """

    def __init__(self, base: nn.Conv1d, r=4, alpha=8.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if r > 0:
            in_c, out_c = base.in_channels, base.out_channels
            k = base.kernel_size[0]
            pad = base.padding[0]
            stride = base.stride[0]
            dilation = base.dilation[0]
            groups = base.groups

            # A: 1x1 pointwise, 保长不变
            self.A = nn.Conv1d(in_c, r, kernel_size=1, bias=False)
            # B: 与 base 超参一致（非常重要：stride/dilation/groups）
            self.B = nn.Conv1d(
                r,
                out_c,
                kernel_size=k,
                padding=pad,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=False,
            )

            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.scaling = alpha / r
            for p in self.base.parameters():
                p.requires_grad = False
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 1.0

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            y = y + self.B(self.A(self.drop(x))) * self.scaling
        return y


class LoRAConvTranspose1d(nn.Module):
    """ LoRA for ConvTranspose1d: base(x) + B(A(x)) """

    def __init__(self, base: nn.ConvTranspose1d, r=4, alpha=8.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if r > 0:
            in_c, out_c = base.in_channels, base.out_channels
            k = base.kernel_size[0]
            pad = base.padding[0]
            stride = base.stride[0]
            dilation = base.dilation[0]
            out_pad = base.output_padding[0]
            groups = base.groups

            # A: 1x1 pointwise Conv1d（先降维）
            self.A = nn.Conv1d(in_c, r, kernel_size=1, bias=False)
            # B: ConvTranspose1d，超参必须与 base 保持一致
            self.B = nn.ConvTranspose1d(
                r,
                out_c,
                kernel_size=k,
                stride=stride,
                padding=pad,
                output_padding=out_pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )

            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.scaling = alpha / r
            for p in self.base.parameters():
                p.requires_grad = False
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 1.0

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            z = self.A(self.drop(x))          # (B, r, L)
            z = self.B(z)                     # (B, out_c, L')
            y = y + z * self.scaling
        return y


def apply_lora_to_model(
    module,
    r=4,
    alpha=8.0,
    dropout=0.0,
    which=("conv_first2", "final_linear"),
    only_generator=False,
):
    """
    递归给目标层套 LoRA。
    - r=0: 不改
    - which:
        "conv_first2": 仅前两层 Conv/ConvT
        "final_linear": 所有 Linear（通常 G/D 的最后几层）
    - only_generator: True 时，只对名字里包含 "generator" 的模块应用（或你在调用处自行控制）
    """
    if r == 0:
        return module

    conv_seen = {"n": 0}

    def _wrap(m, prefix=""):
        for name, child in list(m.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            _wrap(child, child_prefix)

            # 如果只想给 Generator 加 LoRA，可以根据命名/类型约束
            if only_generator and not child_prefix.lower().startswith("generator"):
                # 只是一种参考方式：你也可以在调用处拆开对 G/D 的调用
                pass

            # Conv1d
            if isinstance(child, nn.Conv1d):
                conv_seen["n"] += 1
                if "conv_first2" in which and conv_seen["n"] <= 2:
                    setattr(m, name, LoRAConv1d(child, r=r, alpha=alpha, dropout=dropout))

            # ConvTranspose1d（Generator 里都是它）
            elif isinstance(child, nn.ConvTranspose1d):
                conv_seen["n"] += 1
                if "conv_first2" in which and conv_seen["n"] <= 2:
                    setattr(m, name, LoRAConvTranspose1d(child, r=r, alpha=alpha, dropout=dropout))

            # Linear（通常用于 D 的 fc、也可能用于 G 的尾层）
            elif isinstance(child, nn.Linear):
                if "final_linear" in which:
                    setattr(m, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    _wrap(module)
    return module
