import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m

N = 10000
h = 10000


# Galaxy 1 parameters
x01 = 0
y01 = 0
z01 = 50
num_ell = 300

sizea1 = 8.1539e20
sizeb1 = 8.0e20
del_a = sizea1 / num_ell
a = sizea1 - (num_ell - 1) * del_a

del_b = sizeb1 / num_ell
b = sizeb1 - (num_ell - 1) * del_b
twist = 8 * np.pi / num_ell

# Galaxy 2 parameters
x02 = 2.5000E21
y02 = 0
z02 = 20
num_ell2 = 400

sizea2 = 1.0337E21
sizeb2 = 1.00E21

del_a2 = sizea2 / num_ell2
a2 = sizea2 - (num_ell2 - 1) * del_a2

del_b2 = sizeb2 / num_ell2
b2 = sizeb2 - (num_ell2 - 1) * del_b2
twist2 = 6 * np.pi / num_ell2

n1 = int(N / num_ell)
n2 = int(N / num_ell2)

def ellipse(n, a, b, x0, y0, z0):
    """
    A simple function to generate an ellipse in 3D space with given parameters.

    Parameters:
    n (int): Number of points on the ellipse.
    a (float): Semi-major axis of the ellipse.
    b (float): Semi-minor axis of the ellipse.
    x0 (float): X-coordinate of the ellipse center.
    y0 (float): Y-coordinate of the ellipse center.
    z0 (float): Z-coordinate of the ellipse center.
    
    This function returns a 2D array representing the ellipse points in 3D space.
    """
    ellipse = np.zeros((n, 3))
    angles = np.linspace(0, 2 * np.pi, n)
    for i in range(n):
        x = a * np.cos(angles[i]) + x0
        y = b * np.sin(angles[i]) + y0
        z = h + z0
        ellipse[i] = np.array([x, y, z])
    return ellipse

def multiple_ellipses(n, a, del_a, b, del_b, num_ell, x0, y0, z0):
    """
    A simple function to generate multiple ellipses in 3D space with a gradual increase 
    in the size of the semi-major and semi-minor axes.

    Parameters:
    n (int): Number of points on each ellipse.
    a (float): Initial semi-major axis of the ellipse.
    del_a (float): Increment for the semi-major axis for each ellipse.
    b (float): Initial semi-minor axis of the ellipse.
    del_b (float): Increment for the semi-minor axis for each ellipse.
    num_ell (int): Number of ellipses.
    x0 (float): X-coordinate of the ellipse center.
    y0 (float): Y-coordinate of the ellipse center.
    z0 (float): Z-coordinate of the ellipse center.
    
    This function returns a 3D array representing multiple ellipses in 3D space.
    """
    ellipses = np.zeros((num_ell, n, 3))
    for j in range(num_ell):
        ellipses[j] = ellipse(n, a + del_a * j, b + del_b * j, x0, y0, z0)
    return ellipses

def twist_ellipses(n, a, del_a, b, del_b, num_ell, twist, x0, y0, z0):
    """
    A simple function to generate twisted ellipses in 3D space by applying a twist to each ellipse.

    Parameters:
    n (int): Number of points on each ellipse.
    a (float): Initial semi-major axis of the ellipse.
    del_a (float): Increment for the semi-major axis for each ellipse.
    b (float): Initial semi-minor axis of the ellipse.
    del_b (float): Increment for the semi-minor axis for each ellipse.
    num_ell (int): Number of ellipses.
    twist (float): Twist angle applied to each ellipse.
    x0 (float): X-coordinate of the ellipse center.
    y0 (float): Y-coordinate of the ellipse center.
    z0 (float): Z-coordinate of the ellipse center.
    
    This function returns a 3D array representing twisted ellipses in 3D space.
    """
    ellipses = multiple_ellipses(n, a, del_a, b, del_b, num_ell, x0, y0, z0)
    for j in range(num_ell):
        for i in range(n):
            x = ellipses[j][i][0]
            y = ellipses[j][i][1]
            ellipses[j][i][0] = (x - x0) * np.cos(twist * j) - (y - y0) * np.sin(twist * j) + x0
            ellipses[j][i][1] = (x - x0) * np.sin(twist * j) + (y - y0) * np.cos(twist * j) + y0
    return ellipses

# Plotting Galaxy 1
twisted_ellipses1 = twist_ellipses(n1, a, del_a, b, del_b, num_ell, twist, x01, y01, z01)

plt.figure(figsize=(10, 10))
plt.xlim(-1e21, 1e21)
plt.ylim(-1e21, 1e21)

for i in range(num_ell):
    for j in range(n1):
        plt.scatter(twisted_ellipses1[i][j][0], twisted_ellipses1[i][j][1], c='blue', s=5)

plt.gca().set_aspect('equal')
plt.show()

# Plotting Galaxy 2
twisted_ellipses2 = twist_ellipses(n2, a2, del_a2, b2, del_b2, num_ell2, twist2, x02, y02, z02)

plt.figure(figsize=(10, 10))
plt.xlim(x02 - 1e21, x02 + 1e21)
plt.ylim(y02 - 1e21, y02 + 1e21)

for i in range(num_ell2):
    for j in range(n2):
        plt.scatter(twisted_ellipses2[i][j][0], twisted_ellipses2[i][j][1], c='red', s=5)

plt.gca().set_aspect('equal')
plt.show()

# 3D Rotated Plot for Galaxy 1
phi = np.pi / 3
theta = 0
psi = 0

def rotation_matrix(phi, theta, psi):
    """
    A simple function to generate a 3D rotation matrix using Euler angles.

    Parameters:
    phi (float): Rotation angle around the X-axis.
    theta (float): Rotation angle around the Y-axis.
    psi (float): Rotation angle around the Z-axis.
    
    This function returns a 3D rotation matrix.
    """
    Rx = np.matrix([[1, 0, 0],
                    [0, m.cos(phi), -m.sin(phi)],
                    [0, m.sin(phi), m.cos(phi)]])
    
    Ry = np.matrix([[m.cos(theta), 0, m.sin(theta)],
                    [0, 1, 0],
                    [-m.sin(theta), 0, m.cos(theta)]])
    
    Rz = np.matrix([[m.cos(psi), -m.sin(psi), 0],
                    [m.sin(psi), m.cos(psi), 0],
                    [0, 0, 1]])
    
    return Rz * Ry * Rx

rotation = rotation_matrix(phi, theta, psi)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_ell):
    rotated_ellipse = rotation * twisted_ellipses1[i].T
    ax.scatter(rotated_ellipse[0, :], rotated_ellipse[1, :], rotated_ellipse[2, :], c='blue', s=1)

ax.set_box_aspect([1, 1, 1])
plt.show()

# 3D Rotated Plot for Galaxy 2
phi2 = np.pi / 6
theta2 = 0
psi2 = 0

rotation2 = rotation_matrix(phi2, theta2, psi2)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(num_ell2):
    rotated_ellipse2 = rotation2 * twisted_ellipses2[i].T
    ax.scatter(rotated_ellipse2[0, :], rotated_ellipse2[1, :], rotated_ellipse2[2, :], c='red', s=1)

ax.set_box_aspect([1, 1, 1])
plt.show()