SUPPORTED_VIT_MODELS = (
    "vit_b_16",
    "vit_l_16",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "deit3_base_patch16_224",
    "deit3_large_patch16_224",
    "deit3_huge_patch14_224",
)

MOBILENET_MODEL = "mobilenetv1_100"

VIT_PRUNE_SCOPES = ("qkv", "qkv_out", "qkv_out_mlp", "mlp", "qk", "qk_mlp")
MOBILENET_PRUNE_SCOPES = ("conv_only", "fc_only", "conv_fc", "conv_dw", "all")
