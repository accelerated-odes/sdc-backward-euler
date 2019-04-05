import numpy as np
import matplotlib.pyplot as plt

nsteps, y0, y1, y2 = np.loadtxt("tol_1e-12.txt", skiprows=3, unpack=True)

def compute_delta(y):
    dy = []
    for i in range(y.size-1):
        dy.append(abs(y[i] - y[-1]))
    dy.append(0.0)
    return np.array(dy)

dy0 = compute_delta(y0)
dy1 = compute_delta(y1)
dy2 = compute_delta(y2)

dynorm = np.sqrt(dy0**2 + dy1**2 + dy2**2)

def compute_order(dy):
    order = [0.0]
    for i in range(1, dy.size-1):
        order.append(-np.log2(dy[i]/dy[i-1]))
    order.append(0.0)
    return np.array(order)

oy0 = compute_order(dy0)
oy1 = compute_order(dy1)
oy2 = compute_order(dy2)
oynorm = compute_order(dynorm)

for n, o0, o1, o2, d0, d1, d2, on, dn in zip(nsteps, oy0, oy1, oy2, dy0, dy1, dy2, oynorm, dynorm):
    print("{} {} {} {} {} {} {} {} {}".format(n, o0, d0, o1, d1, o2, d2, on, dn))

print("---")

p = []

for i in range(1, nsteps.size-1):
    p.append(np.log10(dynorm[i]/dynorm[i-1])/np.log10(nsteps[i-1]/nsteps[i]))

print(p)

fig = plt.figure()
ax = fig.add_subplot(111)

time_steps = 1.0/nsteps

ax.set_xlabel("dt")
ax.set_ylabel("order")
ax.plot(time_steps[:-2], p)
fig.savefig("test_sdc_convergence_c.png")

