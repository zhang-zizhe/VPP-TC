import roboticstoolbox as rtb
import numpy as np
import functions as f

# 1) Load the 2-DOF planar robot
robot = rtb.models.DH.Planar2()

# 2) Assign dynamic parameters on each RevoluteDH link:
#    links[0] is joint 1 → link 1;  links[1] is joint 2 → link 2
robot.links[0].m = 2.0                     # mass in kg
robot.links[0].r = [0.5, 0, 0]             # COM at 0.5 m along x
robot.links[0].I = [0.02, 0.02, 0.01, 0, 0, 0]  # [Ixx,Iyy,Izz,Ixy,Ixz,Iyz]

robot.links[1].m = 1.5
robot.links[1].r = [0.4, 0, 0]
robot.links[1].I = [0.015, 0.015, 0.0075, 0, 0, 0]

# 3) Define joint limits:
robot.qmin = np.array([robot.qlim[0, 0], robot.qlim[0, 1]])
robot.qmax = np.array([robot.qlim[1, 0], robot.qlim[1, 1]])
robot.qdlim   = np.array([1.0, 1.2])    # max joint speeds (rad/s)
robot.qddlim  = np.array([5.0, 6.0])    # max joint accels (rad/s²)
robot.tau_lim = np.array([10.0, 8.0])   # max joint torques (Nm)

# # 4) Inspect what you’ve set:
# print("Link dynamics:")
# for i, L in enumerate(robot.links, 1):
#     print(f" • Link {i}: m={L.m} kg,  COM={L.r},  I={L.I}")
# print("\nJoint limits:")
# print(" qlim  =", robot.qlim)
# print(" qdlim  =", robot.qdlim)
# print(" qddlim =", robot.qddlim)
# print(" tau_lim=", robot.tau_lim)
# print(robot.qmin, robot.qmax)
dt = 0.02

# prepare a container for all the “good” samples
results = []

for _ in range(10):
    # sample q, qd near their limits
    robot.q  = f.random_pos_near_limits(robot.qmin, robot.qmax, margin=0.2)
    robot.qd = f.random_vel_near_limits(robot.qdlim,   margin=0.2)

    # compute accel bounds
    qdd_lb, qdd_ub = f.compute_joint_acceleration_bounds_vec(
        robot.q, robot.qd,
        robot.qmin, robot.qmax,
        robot.qdlim, robot.qddlim,
        dt
    )

    # check feasibility of every joint
    if np.any(qdd_lb > qdd_ub):
        # infeasible—skip torque calc and go to next sample
        continue

    # compute torque bounds via RNE
    tau_lb = robot.rne(robot.q, robot.qd, qdd_lb)
    tau_ub = robot.rne(robot.q, robot.qd, qdd_ub)

    # store everything
    results.append({
        'q'      : robot.q.copy(),
        'qd'     : robot.qd.copy(),
        'qdd_lb' : qdd_lb.copy(),
        'qdd_ub' : qdd_ub.copy(),
        'tau_lb' : tau_lb.copy(),
        'tau_ub' : tau_ub.copy(),
    })

# At the end, `results` is a list of dicts for all feasible samples:
print(f"Kept {len(results)} feasible samples out of 10")
for entry in results:
    print(entry)

