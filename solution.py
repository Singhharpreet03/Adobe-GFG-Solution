import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from itertools import combinations

# Helper functions for curve fitting
def fit_circle(XY):
    """Fit a circle to given set of points."""
    def calc_R(xc, yc):
        return np.sqrt((XY[:, 0] - xc) ** 2 + (XY[:, 1] - yc) ** 2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center, _ = curve_fit(f, (0, 0), np.zeros(XY.shape[0]))
    R = calc_R(*center).mean()
    return center, R

def fit_ellipse(XY):
    """Fit an ellipse to given set of points."""
    x = XY[:, 0]
    y = XY[:, 1]
    xmean = np.mean(x)
    ymean = np.mean(y)
    x = x - xmean
    y = y - ymean
    U, S, Vt = np.linalg.svd(np.stack((x, y)))
    tt = np.linspace(0, 2 * np.pi, len(XY))
    ellipse = np.stack((S[0] * np.cos(tt), S[1] * np.sin(tt)))
    ellipse = np.dot(Vt.T, ellipse).T
    ellipse[:, 0] += xmean
    ellipse[:, 1] += ymean
    return ellipse

def is_circle(XY, tolerance=0.01):
    """Check if the given points form a circle."""
    center, R = fit_circle(XY)
    dists = np.sqrt((XY[:, 0] - center[0]) ** 2 + (XY[:, 1] - center[1]) ** 2)
    return np.all(np.abs(dists - R) < tolerance)

def is_straight_line(XY, tolerance=0.01):
    """Check if the given points form a straight line."""
    p1, p2 = XY[0], XY[-1]
    distances = np.abs(np.cross(p2 - p1, XY - p1) / np.linalg.norm(p2 - p1))
    return np.all(distances < tolerance)

def is_ellipse(XY, tolerance=0.01):
    """Check if the given points form an ellipse."""
    ellipse = fit_ellipse(XY)
    diff = np.linalg.norm(XY - ellipse, axis=1)
    return np.all(diff < tolerance)

def is_polygon(XY, tolerance=0.01):
    """Check if the given points form a regular polygon."""
    hull = ConvexHull(XY)
    angles = []
    for i in range(len(hull.vertices)):
        p1, p2, p3 = XY[hull.vertices[i-2]], XY[hull.vertices[i-1]], XY[hull.vertices[i]]
        v1 = p2 - p1
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    angles = np.array(angles)
    return np.all(np.abs(angles - angles.mean()) < tolerance)

def is_star_shape(XY, tolerance=0.01):
    """Check if the given points form a star shape."""
    # Assumes the shape is convex for simplicity
    if is_polygon(XY, tolerance):
        center = np.mean(XY, axis=0)
        dists = np.linalg.norm(XY - center, axis=1)
        return len(set(dists)) > 2  # More than 2 different distances indicates a star
    return False

def plot_paths(paths_XYs):
    """Visualize the paths."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Example usage
# Load the input CSV file paths (polylines)
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Implementing the curve regularization
def regularize_paths(paths_XYs):
    """Regularize the paths by identifying simple geometric shapes."""
    regularized_paths = []
    for XYs in paths_XYs:
        new_paths = []
        for XY in XYs:
            if is_straight_line(XY):
                print("Detected a straight line.")
                new_paths.append(XY)
            elif is_circle(XY):
                print("Detected a circle.")
                center, R = fit_circle(XY)
                theta = np.linspace(0, 2 * np.pi, len(XY))
                circle_XY = np.vstack([center[0] + R * np.cos(theta),
                                       center[1] + R * np.sin(theta)]).T
                new_paths.append(circle_XY)
            elif is_ellipse(XY):
                print("Detected an ellipse.")
                ellipse_XY = fit_ellipse(XY)
                new_paths.append(ellipse_XY)
            elif is_polygon(XY):
                print("Detected a regular polygon.")
                new_paths.append(XY)  # Keep as is, could regularize further
            elif is_star_shape(XY):
                print("Detected a star shape.")
                new_paths.append(XY)  # Keep as is, could regularize further
            else:
                new_paths.append(XY)  # No regularization
        regularized_paths.append(new_paths)
    return regularized_paths

# Symmetry detection
def detect_symmetry(XY, tolerance=0.01):
    """Detect symmetry in the given set of points."""
    symmetries = []
    center = np.mean(XY, axis=0)
    for (i, j) in combinations(range(len(XY)), 2):
        p1, p2 = XY[i], XY[j]
        midpoint = (p1 + p2) / 2
        if np.linalg.norm(midpoint - center) < tolerance:
            symmetries.append((p1, p2))
    return symmetries

# Function to complete incomplete curves
def complete_curves(paths_XYs):
    """Complete any incomplete curves based on provided paths."""
    completed_paths = []
    for XYs in paths_XYs:
        new_paths = []
        for XY in XYs:
            if len(XY) < 2:
                continue
            # Advanced curve completion logic
            if not np.allclose(XY[0], XY[-1]):
                # Interpolation to close the curve
                f_x = interp1d(np.arange(len(XY)), XY[:, 0], kind='cubic', fill_value="extrapolate")
                f_y = interp1d(np.arange(len(XY)), XY[:, 1], kind='cubic', fill_value="extrapolate")
                extended_X = np.arange(len(XY), len(XY) + 5)  # Add 5 more points
                XY_new = np.vstack([f_x(extended_X), f_y(extended_X)]).T
                XY = np.vstack([XY, XY_new, XY[0]])  # Close the curve
            new_paths.append(XY)
        completed_paths.append(new_paths)
    return completed_paths

# Example workflow
paths_XYs = read_csv('examples/isolated.csv')
regularized_paths = regularize_paths(paths_XYs)
plot_paths(regularized_paths)

completed_paths = complete_curves(paths_XYs)
plot_paths(completed_paths)

# Detect symmetry in the regularized paths
for XYs in regularized_paths:
    for XY in XYs:
        symmetries = detect_symmetry(XY)
        print(f"Detected {len(symmetries)} symmetries.")
