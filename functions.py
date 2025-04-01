def acc_bounds_from_pos(q, qd, qmin, qmax, dt):
    qddmax1 = -qd/dt
    qddmax2 = -qd**2/(2*(qmax-q))
    qddmax3 = 2*(qmax-q-dt*qd)/(dt**2)
    qddmin2 = qd**2/(2*(q-qmin))
    qddmin3 = 2 * (qmin - q - dt * qd) / (dt**2)

    if qd >= 0:
        qdd_lb = qddmin3
        if qddmax3 > qddmax1:
            qdd_ub = qddmax3
        else:
            qdd_ub = min(qddmax1, qddmax2)
    else:
        qdd_ub = qddmax3
        if qddmin3 < qddmax1:
            qdd_lb = qddmin3
        else:
            qdd_lb = max(qddmax1, qddmin2)

    return qdd_lb, qdd_ub