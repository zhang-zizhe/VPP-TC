import pybullet as p
import pybullet_data
import time

# 1. 以 GUI 模式连接 PyBullet（会弹出一个可视化窗口）
physicsClient = p.connect(p.GUI)

# 2. 可选：添加重力，让模型在初始化时“settle”一下
p.setGravity(0, 0, -9.81)

# 3. 告诉 PyBullet 去哪里找常见的 URDF 文件（此处使用内置的示例路径）
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 4. 加载 Franka（Panda）URDF，开启自碰撞检测
franka_id = p.loadURDF(
    "panda/panda.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
)
p.setCollisionFilterPair(franka_id, franka_id, 4, 6, enableCollision=0)
# 5. 将 7 个关节设置到你想测试的弧度角度（radian）
joints = [0, 1, 2, 3, 4, 5, 6]
target_positions = [-0.756, 1.19, 2.35, -2.09, -1.99, 0.149, -2.22]
for i, q in zip(joints, target_positions):
    p.resetJointState(franka_id, i, q)

# 6. 多跑几个 step，让引擎计算出碰撞并且 GUI 能渲染出来
for _ in range(10):
    p.stepSimulation()
    time.sleep(1/240)

# 7. 查询同一个 body（franka_id）内部的碰撞
contacts = p.getContactPoints(bodyA=franka_id, bodyB=franka_id)
if len(contacts) == 0:
    print("✅ No self‐collision detected.")
    print("按 Ctrl+C 或关闭窗口结束查看。")
    # 保持仅读循环，让 GUI 不会立即退出
    while True:
        p.stepSimulation()
        time.sleep(1/240)
else:
    print(f"❌ {len(contacts)} contact(s) found. Drawing debug markers at each contact point...")
    for c in contacts:
        # 不同 PyBullet 版本，c[5] 可能是 (x,y,z) 三元组，也可能是单独的 float
        if isinstance(c[5], (tuple, list)) and len(c[5]) == 3:
            contact_xyz = list(c[5])  # 已经是 (x,y,z)
        else:
            # 如果 c[5],c[6],c[7] 本身就是三个 float
            contact_xyz = [c[5], c[6], c[7]]

        # 在 contact_xyz 处画三条短线 (分别沿 X/Y/Z 方向)，颜色分别为 红、绿、蓝
        p.addUserDebugLine(
            contact_xyz,
            [contact_xyz[0] + 0.02, contact_xyz[1], contact_xyz[2]],
            [1, 0, 0],    # 红色
            lineWidth=2,
            lifeTime=0    # 0 表示永久可见，直到断开连接或手动清除
        )
        p.addUserDebugLine(
            contact_xyz,
            [contact_xyz[0], contact_xyz[1] + 0.02, contact_xyz[2]],
            [0, 1, 0],    # 绿色
            lineWidth=2,
            lifeTime=0
        )
        p.addUserDebugLine(
            contact_xyz,
            [contact_xyz[0], contact_xyz[1], contact_xyz[2] + 0.02],
            [0, 0, 1],    # 蓝色
            lineWidth=2,
            lifeTime=0
        )

    print("已在碰撞位置画出彩色小十字标记，GUI 中可见。")
    print("用鼠标缩放/旋转视角，查看是哪两个连杆相互碰撞。")
    # 保持仅读循环，这样你可以在 GUI 里自由导航
    while True:
        p.stepSimulation()
        time.sleep(1/240)

# （此段代码通常无法执行到，因为我们用无限循环来保持 GUI 打开）
p.disconnect()
