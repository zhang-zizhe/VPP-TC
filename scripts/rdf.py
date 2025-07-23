import torch
from RDF.bf_sdf import BPSDF
from RDF.panda_layer.panda_layer import PandaLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'RDF/models/BP_24.pt'
sdf_model = torch.load(model_path, map_location=device, weights_only=False)

# 初始化 PandaLayer & BPSDF
panda_layer = PandaLayer(device)
bpsdf = BPSDF(
    n_func=24,
    domain_min=-1.0,
    domain_max=1.0,
    robot=panda_layer,
    device=device
)


import numpy as np

def query_sdf(x_np: np.ndarray,
                         pose_np: np.ndarray,
                         theta_np: np.ndarray):
    """
    查询 SDF 距离、最近连杆索引，并计算 dst 对 q 的梯度。

    Args:
        x_np:     (3,) numpy array
        pose_np:  (4,4) numpy array
        theta_np: (7,) numpy array

    Returns:
        dst       (float): SDF 距离
        link_id   (int):   最近连杆索引
        grad_q    (np.ndarray shape (7,)): ∂dst/∂q
    """
    # 1) 转 numpy→tensor，theta 要 requires_grad
    x_t     = torch.from_numpy(x_np.reshape(1,3).astype(np.float32)).to(device)
    pose_t  = torch.from_numpy(pose_np.reshape(1,4,4).astype(np.float32)).to(device)
    theta_t = torch.from_numpy(theta_np.reshape(1,7).astype(np.float32)) \
                   .to(device).requires_grad_(True)

    # 2) 前向计算
    sdf_vals, _, idx = bpsdf.get_whole_body_sdf_batch(
        x_t, pose_t, theta_t, sdf_model,
        use_derivative=False,   # 这里关闭 BPSDF 自带的导数输出
        return_index=True
    )
    # squeeze 成标量
    sdf_val = sdf_vals.squeeze()   # shape=()

    # 3) 反向传播
    # 清空可能存在的旧梯度
    if theta_t.grad is not None:
        theta_t.grad.zero_()
    sdf_val.backward()             # 计算 ∂sdf_val/∂theta_t

    # 4) 拷贝到 numpy
    dst     = float(sdf_val.item())
    link_id = int(idx.item())
    grad_q  = theta_t.grad.detach().cpu().numpy().reshape(-1)  # (7,)

    return dst, link_id, grad_q
