import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.utils import make_grid


def pca_latent(lat: torch.Tensor) -> torch.Tensor:
    if len(lat.shape) == 3:
        lat = lat.unsqueeze(0)
    if lat.shape[1] == 3:
        return lat

    B, C, W, H = lat.shape
    lat_reshaped = lat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

    pca = PCA(n_components=3)
    lat_pca = pca.fit_transform(lat_reshaped)

    lat_pca = torch.from_numpy(lat_pca).float().reshape(B, W, H, 3).permute(0, 3, 1, 2).to(lat.device)
    return lat_pca


def plot_diffusion(x, nrow=16, figsize=(20, 20)):
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(
        make_grid(
            x,
            nrow=nrow,
            normalize=True,
        ).permute(1, 2, 0)
    )
    plt.show()


def image_grid(imgs, rows, border_size=0, resize_factor=1):
    cols = len(imgs) // rows

    w, h = imgs[0].size
    w, h = int(w * resize_factor), int(h * resize_factor)
    bordered_w, bordered_h = w + 2 * border_size, h + 2 * border_size

    grid = Image.new("RGB", size=(cols * bordered_w, rows * bordered_h), color=(255, 255, 255))

    for i, img in enumerate(imgs):
        img = img.resize((w, h))
        img_with_border = ImageOps.expand(img, border=border_size, fill="black")
        grid.paste(img_with_border, box=(i % cols * bordered_w, i // cols * bordered_h))
    return grid


def tensors_to_pils_single(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return transforms.ToPILImage()(tensor)


def grid_tensors(tensors, rows: int = 1, border_size: int = 0, resize_factor: int = 1):
    assert len(tensors.shape) == 4, f"tensors must have 4 dimensions, got {tensors.shape}"
    return image_grid(
        [tensors_to_pils_single(i_tens) for i_tens in tensors],
        rows=rows,
        border_size=border_size,
        resize_factor=resize_factor,
    )
