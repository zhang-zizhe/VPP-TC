import torch
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "RDF", "models", "BP_24.pt")
from RDF.bf_sdf import BPSDF
from RDF.panda_layer.panda_layer import PandaLayer
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sdf_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

panda_layer = PandaLayer(device)
bpsdf = BPSDF(
    n_func=24,
    domain_min=-1.0,
    domain_max=1.0,
    robot=panda_layer,
    device=device
)

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

    # # 2) 前向计算
    # sdf_vals, _, idx = bpsdf.get_whole_body_sdf_batch(
    #     x_t, pose_t, theta_t, sdf_model,
    #     use_derivative=False,   # 这里关闭 BPSDF 自带的导数输出
    #     return_index=True
    # )
    # # squeeze 成标量
    # sdf_val = sdf_vals.squeeze()   # shape=()

    # # 3) 反向传播
    # # 清空可能存在的旧梯度
    # if theta_t.grad is not None:
    #     theta_t.grad.zero_()
    # sdf_val.backward()             # 计算 ∂sdf_val/∂theta_t

    # # 4) 拷贝到 numpy
    # dst     = float(sdf_val.item())
    # link_id = int(idx.item())
    # grad_q  = theta_t.grad.detach().cpu().numpy().reshape(-1)  # (7,)

    # return dst, grad_q#, link_id
    with torch.no_grad():
        sdf, d_sdf = bpsdf.get_whole_body_sdf_with_joints_grad_batch(
            x_t, pose_t, theta_t, sdf_model
        )
        grad_q = d_sdf.squeeze(0).squeeze(0).cpu().numpy()  # (7,)
        sdf_val = sdf.squeeze()
        dst     = float(sdf_val.item())

    return dst, grad_q

def query_sdf_batch(x_np: np.ndarray,
                    pose_np: np.ndarray,
                    theta_np: np.ndarray,
                    need_index: bool = False):
    # -------- 1) 规范化输入形状 --------
    x_np = np.asarray(x_np, dtype=np.float32)
    # 关键：无论进来是 (3,), (1,3), (2,1,3) 还是 (N,3)，全部折成 (N,3)
    if x_np.size % 3 != 0:
        raise ValueError("x_np 的元素个数必须是 3 的倍数")
    x_np = x_np.reshape(-1, 3)   # <<< 这行解决 (2,1,3) 的问题

    theta_np = np.asarray(theta_np, dtype=np.float32)
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, 7)
    elif theta_np.shape[-1] != 7:
        raise ValueError("theta_np 应为 (...,7)")

    B = theta_np.shape[0]
    N = x_np.shape[0]

    pose_np = np.asarray(pose_np, dtype=np.float32)
    if pose_np.shape == (4, 4):
        pose_np = np.broadcast_to(pose_np, (B, 4, 4))
    elif pose_np.shape != (B, 4, 4):
        raise ValueError("pose_np 形状应为 (4,4) 或 (B,4,4)")

    # -------- 2) 转 tensor --------
    x_t     = torch.from_numpy(x_np).to(device)       # (N,3)
    pose_t  = torch.from_numpy(pose_np).to(device)    # (B,4,4)
    theta_t = torch.from_numpy(theta_np).to(device)   # (B,7)

    # -------- 3) 前向 --------
    with torch.no_grad():
        sdf0, d_sdf = bpsdf.get_whole_body_sdf_with_joints_grad_batch(
            x_t, pose_t, theta_t, sdf_model
        )
        dsts    = sdf0.detach().cpu().numpy().astype(np.float32)   # (B,N)
        grad_qs = d_sdf.detach().cpu().numpy().astype(np.float32)  # (B,N,7)

    if not need_index:
        return dsts, grad_qs

    with torch.no_grad():
        sdf_vals, _, idx = bpsdf.get_whole_body_sdf_batch(
            x_t, pose_t, theta_t, sdf_model,
            use_derivative=False, return_index=True
        )
        link_ids = idx.detach().cpu().numpy().astype(np.int32)     # (B,N)

    return dsts, grad_qs, link_ids

