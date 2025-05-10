#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import numpy as np
import csv

def main():
    # 连接物理仿真（可改为 p.GUI 观察）
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 加载机器人 URDF（这里以 Panda 为例，可替换为你的 URDF 路径）
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF("panda/panda.urdf",
                    useFixedBase=True,
                    flags=flags)

    # 获取所有可动关节的索引及限位
    joint_indices = []
    joint_limits = []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            joint_indices.append(i)
            joint_limits.append((info[8], info[9]))  # lower, upper

    # 采样设置
    n_samples = 100000
    samples = []
    collision_flags = []
    N_collision = 0

    for idx in range(n_samples):
        # 随机生成一组关节角度
        q = [np.random.uniform(low, high) for (low, high) in joint_limits]
        samples.append(q)

        # 设定机器人到该关节配置
        for joint, angle in zip(joint_indices, q):
            p.resetJointState(robot, joint, angle)
        p.stepSimulation()

        # 检测是否有自碰撞
        contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision = len(contacts) > 0
        if collision:
            N_collision+=1
        collision_flags.append(collision)

        if (idx+1) % 10000 == 0:
            print(f"Sample {idx}: collision = {N_collision}")

    # 将结果写入 CSV
    with open("collision_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"joint_{i}" for i in range(len(joint_indices))] + ["collision"]
        writer.writerow(header)
        for q, flag in zip(samples, collision_flags):
            writer.writerow(q + [int(flag)])

    p.disconnect()

if __name__ == "__main__":
    main()
