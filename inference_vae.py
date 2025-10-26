import argparse
import numpy as np
import torch
import trimesh

from triposg.inference_utils import hierarchical_extract_geometry
from triposg.models.autoencoders import TripoSGVAEModel
from huggingface_hub import snapshot_download

import sys
sys.path.append('./')


def load_surface(data_path, num_pc=204800):
    data = np.load(data_path, allow_pickle=True)
    print(type(data))
    data = data.tolist()
    print(type(data))
    print(data.keys())
    print(data["surface_points"])
    print(data["surface_normals"])
    surface = data["surface_points"]  # Nx3
    normal = data["surface_normals"]  # Nx3
    print(surface.shape)
    print(normal.shape)

    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_pc, replace=False)
    print(ind.shape)
    print(ind)
    # Convert the randomly sampled surface points from numpy array to PyTorch tensor
    # surface[ind] selects the sampled points using the random indices
    surface = torch.FloatTensor(surface[ind])
    print(surface.shape)
    print(surface)

    normal = torch.FloatTensor(normal[ind])
    # 将表面点坐标和法向量拼接，并调整维度后移至GPU
    # torch.cat([surface, normal], dim=-1): 在最后一个维度上拼接两个张量
    #   - surface: [204800, 3] 表面点的xyz坐标
    #   - normal: [204800, 3] 对应点的法向量
    #   - 拼接后: [204800, 6] 每个点包含6个值(x,y,z,nx,ny,nz)
    surface = torch.cat([surface, normal], dim=-1)
    print(surface.shape)
    print(surface)
    # .unsqueeze(0): 在第0维增加一个维度，用于添加batch维度
    #   - 从 [204800, 6] 变为 [1, 204800, 6]
    #   - 1表示batch_size为1，即一次处理一个样本
    # .cuda(): 将张量移动到GPU显存中进行加速计算
    surface = surface.unsqueeze(0).cuda()
    print(surface.shape)
    print(surface)


    return surface


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-input", type=str, required=True)
    args = parser.parse_args()

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)

    vae: TripoSGVAEModel = TripoSGVAEModel.from_pretrained(
        triposg_weights_dir,
        subfolder="vae",
    ).to(device, dtype=dtype)

    # load surface from sdf and encode
    surface = load_surface(
        args.surface_input, num_pc=204800
    ).to(device, dtype=dtype)
    sample = vae.encode(surface).latent_dist.sample()
    
    # vae infer 
    with torch.no_grad():
        geometric_func = lambda x: vae.decode(sample, sampled_points=x).sample
        output = hierarchical_extract_geometry(
            geometric_func,
            device,
            bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
            dense_octree_depth=8,
            hierarchical_octree_depth=9,
        )
        meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]

    meshes[0].export("test_vae.glb")

