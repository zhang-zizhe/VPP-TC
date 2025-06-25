import os, torch, numpy as np, pybullet as p, pybullet_data, time
from panda_layer.panda_layer import PandaLayer
from bf_sdf import BPSDF          # >>> 确保 import 路径正确
# -------------------- 1.  BP-24 网络 -----------------------------
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_ckpt= "BP_24.pt"
bp_model= torch.load(net_ckpt, map_location=device)

panda_layer = PandaLayer(device)
bpsdf = BPSDF(n_func=24, domain_min=-1.0, domain_max=1.0,
              robot=panda_layer, device=device)

# 查询点 & 关节角
x_query = torch.tensor([[0.25, -0.25, 0.30]], device=device)          # (1,3)
theta   = torch.tensor([[-1.372 , -0.5767, -0.7912,
                         -2.1332, -0.2909,  2.1742,  1.9866]], device=device)
world_T_base = torch.eye(4, device=device).unsqueeze(0)               # (1,4,4)

with torch.no_grad():
    sdf_val, _, link_idx = bpsdf.get_whole_body_sdf_batch(
        x_query, world_T_base, theta, bp_model,
        use_derivative=False, return_index=True
    )

bp_dist   = sdf_val.item()
bp_linkID = link_idx.item()

# -------------------- 2.  PyBullet 真距离 -------------------------
p.connect(p.DIRECT)                              # 无 GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF(
    "models/urdf/panda/panda.urdf",
    useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION
)
# 把关节角塞进去
for j, q in enumerate(theta.squeeze().cpu().numpy()):
    p.resetJointState(robot_id, j, q)

# 逐个连杆 (0-6) 计算点到该连杆表面的最近距离
min_dist_pb, min_link_pb = np.inf, -1
pt = x_query.squeeze().cpu().numpy()
for link in range(7):
    res = p.getClosestPoints(
        bodyA=robot_id, bodyB=-1,          # 固定世界坐标
        distance=1000.0,                   # 足够大; 取最近
        linkIndexA=link,
        collisionShapeA=-1,
        # 对点用障碍物技巧: create a temporary sphere as bodyB = -1
        # 但 pybullet 支持 “point vs mesh” 直接传 -1
        physicsClientId=0
    )
    # res 可能为空 => 无碰撞; 若不为空取最近
    if res:
        d = min([cp[8] for cp in res])     # 第 9 个字段 = 距离
        if d < min_dist_pb:
            min_dist_pb, min_link_pb = d, link

p.disconnect()

# -------------------- 3. 结果比较 -------------------------------
print("\n=== BP-24 预测 ===")
print(f"  最近连杆 ID  : {bp_linkID}")
print(f"  预测距离     : {bp_dist:.6f} m")

print("\n=== PyBullet 实测 ===")
print(f"  最近连杆 ID  : {min_link_pb}")
print(f"  实际距离     : {min_dist_pb:.6f} m")

print("\n=== 误差 ===")
print(f"  Δdistance    : {abs(bp_dist - min_dist_pb):.6f} m")
print()

