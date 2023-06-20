
# --------------------------------------------------------------------------------
#
#                        New Methods in Computational Physics
#
#                                N - Body Simulation
#
# -----------------------                             ----------------------------
# Arnaud Remi - S183416 |                             | Victor Mangeleer - S181670
# --------------------------------------------------------------------------------
#
# ---------------------------
#         Librairies
# ---------------------------
import os
import imageio
import numpy   as np

# ---------------------------
#   Simulation parameters
# ---------------------------
# Define size of the border to crop vizualization gif (x and y direction)
Bx = 0.05
By = 0.05

# Resolution of the gif
resolution = 256

# ---------------------------
#          Functions
# ---------------------------
# Used to load numpy data
def load_array_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    arr = []
    for line in lines:
        row = [float(x) for x in line.strip().split()]
        arr.append(row)
    return np.array(arr)

# ---------------------------
#           Main
# ---------------------------
if __name__ == '__main__':

    # Loading data
    x  = load_array_from_file("simulations/position_x.txt")
    y  = load_array_from_file("simulations/position_y.txt")
    vx = load_array_from_file("simulations/speed_x.txt")
    vy = load_array_from_file("simulations/speed_y.txt")

    # Shifting data
    x0_min = np.min(x[:,0])
    x0_max = np.max(x[:,0])
    y0_min = np.min(y[:,0])
    y0_max = np.max(y[:,0])

    x = (x - x0_min + Bx * (x0_min + x0_max - 2 * x)) / (x0_max - x0_min)
    y = (y - y0_min + By * (y0_min + y0_max - 2 * y)) / (y0_max - y0_min)

    # Intermediate variables
    nb_bodies  = x.shape[0]
    nb_iters   = x.shape[1]

    # Transformation to index
    x = (x * resolution).astype(int)
    y = (y * resolution).astype(int)

    # Stores the results in numpy matrix format
    # simulation = np.ones((nb_iters, resolution + 1, resolution + 1)) * 255
    simulation = np.ones((nb_iters, resolution + 1, resolution + 1))

    # Loading the simulation with results
    for i in range(nb_iters):
        for b in range(nb_bodies):
            xi, yi = x[b][i], y[b][i]
            if xi <= resolution and xi >= 0 and yi <= resolution and yi >= 0.0:
                # simulation[i][xi][yi] = 0
                simulation[i][xi][yi] = 255

    # Conversion to BW format
    simulation = simulation.astype(np.uint8)

    # Total number of gifs in the benchmark folder
    nb_file = len(os.listdir("gifs"))

    # Write the array to the output file as a GIF
    imageio.mimsave(f"gifs/simulation_{nb_file}.gif", simulation)