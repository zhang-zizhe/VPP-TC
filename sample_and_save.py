import roboticstoolbox as rtb
import numpy as np
import csv
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

for _ in range(10000):
    # sample q, qd near their limits
    robot.q  = f.random_pos_near_limits(robot.qmin, robot.qmax, margin=0.1)
    robot.qd = f.random_vel_near_limits(robot.qdlim,   margin=0.1)

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
        # print("qdd error")
        continue

    # compute torque bounds via RNE
    tau_lb = robot.rne(robot.q, robot.qd, qdd_lb)
    tau_ub = robot.rne(robot.q, robot.qd, qdd_ub)

    # check feasibility of every joint
    if np.any(tau_lb > tau_ub):
        # infeasible—skip torque calc and go to next sample
        print("tau error")
        continue

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
# print(f"Kept {len(results)} feasible samples out of 10")
# for entry in results:
#     print(entry)

if results:
    print(f"\nPreparing to save {len(results)} samples.")
    # Convert list of dictionaries to a dictionary of lists/arrays for easier saving
    data_for_saving = {}
    keys = results[0].keys() # Get keys from the first sample
    for key in keys:
        data_for_saving[key] = np.array([sample[key] for sample in results])

    # --- Option 1: Save to .npz file (Recommended for NumPy arrays) ---
    npz_filename = "robot_dynamics_samples.npz"
    try:
        np.savez_compressed(npz_filename, **data_for_saving)
        print(f"Data successfully saved to {npz_filename}")
        # To load this data later:
        # loaded_data = np.load(npz_filename)
        # q_data = loaded_data['q']
        # qd_data = loaded_data['qd']
        # ... and so on for other keys.
        # print(f"Example: Shape of loaded 'q' data: {q_data.shape}")
    except Exception as e:
        print(f"Error saving data to {npz_filename}: {e}")

    # --- Option 2: Save to CSV file ---
    csv_filename = "robot_dynamics_samples.csv"
    try:
        # Create header for CSV. For arrays like 'q', it will be 'q_0', 'q_1', ...
        header = []
        first_sample_dict = results[0] # Use the first dictionary to determine sub-column names
        for key in keys:
            value = first_sample_dict[key]
            if isinstance(value, np.ndarray) and value.ndim > 0 : # Check if it's a non-scalar array
                for i in range(value.shape[0]): # Assuming 1D array for each joint property
                    header.append(f"{key}_{i}")
            else: # Scalar or single value
                header.append(key)

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header) # Write the header
            for sample_dict in results: # Iterate through each dictionary in the results list
                row_to_write = []
                for key in keys:
                    value = sample_dict[key]
                    if isinstance(value, np.ndarray) and value.ndim > 0:
                        row_to_write.extend(value.tolist()) # Flatten array to list and extend row
                    else:
                        row_to_write.append(value) # Append scalar value
                writer.writerow(row_to_write)
        print(f"Data successfully saved to {csv_filename}")
        # To load with pandas:
        # import pandas as pd
        # df = pd.read_csv(csv_filename)
        # print(df.head())
    except Exception as e:
        print(f"Error saving data to {csv_filename}: {e}")

else:
    print("No feasible samples were generated. Nothing to save.")