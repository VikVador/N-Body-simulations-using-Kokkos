/*
--------------------------------------------------------------------------------

                        New Methods in Computational Physics

                                N - Body Simulation

-----------------------                             ----------------------------
Arnaud Remi - S183416 |                             | Victor Mangeleer - S181670
--------------------------------------------------------------------------------

*/

#include <limits>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>

//------------------------------------------------------------------------------
//                                 Structure
//------------------------------------------------------------------------------
typedef struct Body
{
    // Mass [Kg]
    double mass;

    // Speed [m/s]
    double vx;
    double vy;

    // Force applied on body [N]
    double fx;
    double fy;

} Body;

//------------------------------------------------------------------------------
//                             Functions - General
//------------------------------------------------------------------------------
// Displays a message and terminates the program
static void terminate(const char *s)
{
    fprintf(stderr, "\n %s \n\n", s);
    exit(-1);
}

//------------------------------------------------------------------------------
//                                  Main
//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    //---------
    // Security
    //---------
    // Checking the number of arguments
    if (argc != 7)
        terminate("ERROR (main) - Wrong number of arguments");

    //-------------------------------------
    // Initialization and loading arguments
    //-------------------------------------
    double maxt   = -1;
    double deltat = -1;
    char *sim_file;

    for (int i = 1; i < argc; i++)
    {
        // Simulation file name
        if (strcmp(argv[i], "-file") == 0)
            sim_file = argv[++i];

        // Total duration
        else if (strcmp(argv[i], "-maxt") == 0)
            maxt = atof(argv[++i]);

        // Time step
        else if (strcmp(argv[i], "-deltat") == 0)
            deltat = atof(argv[++i]);

        // Documentation (if error)
        else
        {
            printf("The N-BODY Simulation should be executed using the following arguments:\n");
            printf("- <file>   : name of the file to load initial conditions from\n");
            printf("- <maxt>   : the total duration, in real time, of the simulation\n");
            printf("- <deltat> : the delta of time (use 0.1 or smaller)\n");
            exit(-1);
        }
    }

    // Opening simulation file
    FILE *sim_data = fopen(sim_file, "rb");

    if (sim_data == NULL)
        terminate("ERROR (fopen) - Unable to open data file");

    // Retrieving total number of particles
    int n_bodies;
    fscanf(sim_data, "%d", &n_bodies);

    // Total number of iterations to make simulation
    int nb_iterations = maxt/deltat;

    // Allocation (1)
    Body *bodies = new Body[n_bodies];
    double **x   = new double *[n_bodies];
    double **y   = new double *[n_bodies];
    double **vx  = new double *[n_bodies];
    double **vy  = new double *[n_bodies];

    // Allocation (2)
    for (int i = 0; i < n_bodies; i++)
    {
        x[i]  = new double[nb_iterations];
        y[i]  = new double[nb_iterations];
        vx[i] = new double[nb_iterations];
        vy[i] = new double[nb_iterations];
    }

    // Loading data from simulation file
    for (int i = 0; i < n_bodies; i++)
    {
        // Initial positions of particles
        fscanf(sim_data, "%lf", &x[i][0]);
        fscanf(sim_data, "%lf", &y[i][0]);

        // Initial velocity & Particle mass
        fscanf(sim_data, "%lf", &bodies[i].vx);
        fscanf(sim_data, "%lf", &bodies[i].vy);
        fscanf(sim_data, "%lf", &bodies[i].mass);

        // Initial force
        bodies[i].fx = 0;
        bodies[i].fy = 0;

        // Initialization of speeds
        vx[i][0] = bodies[i].vx;
        vy[i][0] = bodies[i].vy;
    }

    // Closing the simulation file
    fclose(sim_data);

    //--------------------------
    //       Simulation
    //--------------------------
    for (int t = 1; t < nb_iterations; t++)
    {
        // Computing forces
        for (int i = 0; i < n_bodies; i++)
        {
            // Simulation Parameters
            double G   = 6.67e-11;
            double EPS = 1e-15;

            // Intermediate variables
            double dx   = 0;
            double dy   = 0;
            double dist = 0;
            double    F = 0;

            // --- Reseting forces ---
            bodies[i].fx = 0;
            bodies[i].fy = 0;

            // Computing forces
            for (int j = 0 ; j < n_bodies; j++)
            {
                if (i != j)
                {
                    // Distances
                    dx    = x[j][t - 1] - x[i][t - 1];
                    dy    = y[j][t - 1] - y[i][t - 1];
                    dist  = sqrt(dx * dx + dy * dy);

                    // Force (1)
                    F = bodies[j].mass / (dist * dist + EPS * EPS);

                    // Adding force increment
                    bodies[i].fx += F * dx / dist;
                    bodies[i].fy += F * dy / dist;
                }
            }

            // Force (2)
            bodies[i].fx = G * bodies[i].mass;
            bodies[i].fy = G * bodies[i].mass;
        }

        // Updating positions and velocities
        for (int i = 0; i < n_bodies; i++)
        {
            // Update of speed
            bodies[i].vx += bodies[i].fx * deltat / bodies[i].mass;
            bodies[i].vy += bodies[i].fy * deltat / bodies[i].mass;

            // Adding speeds
            vx[i][t] = bodies[i].vx;
            vy[i][t] = bodies[i].vy;

            // Update of positions
            x[i][t] = x[i][t - 1] + deltat * bodies[i].vx;
            y[i][t] = y[i][t - 1] + deltat * bodies[i].vy;
        }
    }

    // File used to save results
    FILE* sim_results_x  = fopen("simulations/position_x.txt", "w");
    FILE* sim_results_y  = fopen("simulations/position_y.txt", "w");
    FILE* sim_results_vx = fopen("simulations/speed_x.txt", "w");
    FILE* sim_results_vy = fopen("simulations/speed_y.txt", "w");

    if (sim_results_x == NULL)
        terminate("ERROR (fopen) - Unable to open data file (x)");

    if (sim_results_y == NULL)
        terminate("ERROR (fopen) - Unable to open data file (y)");

    if (sim_results_vx == NULL)
        terminate("ERROR (fopen) - Unable to open data file (vx)");

    if (sim_results_vy == NULL)
        terminate("ERROR (fopen) - Unable to open data file (vy)");

    // Saving results
    for(int i = 0; i < n_bodies ; i++)
    {
        for(int t = 0 ; t < nb_iterations ; t++)
        {
            fprintf(sim_results_x,  "%lf ",  x[i][t]);
            fprintf(sim_results_y,  "%lf ",  y[i][t]);
            fprintf(sim_results_vx, "%lf ", vx[i][t]);
            fprintf(sim_results_vy, "%lf ", vy[i][t]);
        }

        fprintf(sim_results_x,  "\n");
        fprintf(sim_results_y,  "\n");
        fprintf(sim_results_vx, "\n");
        fprintf(sim_results_vy, "\n");
    }

    // Finalization
    fclose(sim_results_x);
    fclose(sim_results_y);
    fclose(sim_results_vx);
    fclose(sim_results_vy);

    // Cleaning up
    for (int i = 0; i < n_bodies; i++)
    {
        delete[] x[i];
        delete[] y[i];
        delete[] vx[i];
        delete[] vy[i];
    }

    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    delete[] bodies;

    return 0;
}
