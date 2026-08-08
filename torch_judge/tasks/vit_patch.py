"""ViT Patch Embedding task."""

TASK = {
    "title": "ViT Patch Embedding",
    "difficulty": "Medium",
    "function_name": "PatchEmbedding",
    "hint": "Reshape image into patches: (B, C, H, W) -> (B, num_patches, C*P*P). Then project with nn.Linear(C*P*P, embed_dim). num_patches = (img_size/patch_size)^2.",
    "tests": [
        {
            "name": "Output shape",
            "code": "\nimport torch, torch.nn as nn\npe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64)\nassert isinstance(pe, nn.Module)\nout = pe(torch.randn(2, 3, 32, 32))\nassert out.shape == (2, 16, 64), f'Shape: {out.shape}, expected (2, 16, 64)'\n"
        },
        {
            "name": "num_patches attribute",
            "code": "\nimport torch\npe = {fn}(img_size=224, patch_size=16, in_channels=3, embed_dim=768)\nassert pe.num_patches == 196, f'num_patches: {pe.num_patches}'\n"
        },
        {
            "name": "Different image sizes",
            "code": "\nimport torch\npe = {fn}(img_size=64, patch_size=16, in_channels=1, embed_dim=32)\nout = pe(torch.randn(1, 1, 64, 64))\nassert out.shape == (1, 16, 32), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\nimport torch\npe = {fn}(img_size=32, patch_size=8, in_channels=3, embed_dim=64)\nx = torch.randn(1, 3, 32, 32, requires_grad=True)\npe(x).sum().backward()\nassert x.grad is not None, 'x.grad is None'\n"
        }
    ]
}
