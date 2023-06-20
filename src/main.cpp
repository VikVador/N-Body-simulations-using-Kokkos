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
#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

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
static void terminate(char *s)
{
    fprintf(stderr, "\n %s \n\n", s);
    exit(-1);
}

//------------------------------------------------------------------------------
//                                  Main
//------------------------------------------------------------------------------
int main(int argc, char** argv)
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

    for(int i = 1; i < argc ; i++)
    {
        // Simulation file name
        if(strcmp(argv[i], "-file") == 0)
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
    FILE* sim_data = fopen(sim_file, "rb");

    if (sim_data == NULL)
        terminate("ERROR (fopen) - Unable to open data file");

    // Retreiving total number of particles
    int n_bodies;
    fscanf(sim_data, "%d", &n_bodies);

    //-------
    // Kokkos
    //-------
    Kokkos::initialize(argc, argv);
    {
        // Define if the program should be run in benchmark mode (save results or not)
        bool benchmark = true;

        // Define if cuda graph should be used or not
        bool use_graph = false;

        // Defining Kokkos possible configurations
        #if defined(ACTIVATE_KOKKOS_SERIAL_LEFT)
            typedef Kokkos::OpenMP     ExecSpace; // OpenMP threads must be set to 1
            typedef Kokkos::HostSpace   MemSpace;
            typedef Kokkos::LayoutLeft    Layout;
        #elif defined(ACTIVATE_KOKKOS_SERIAL_RIGHT)
            typedef Kokkos::OpenMP     ExecSpace; // OpenMP threads must be set to 1
            typedef Kokkos::HostSpace   MemSpace;
            typedef Kokkos::LayoutRight   Layout;
        #elif defined(ACTIVATE_KOKKOS_OPENMP_LEFT)
            typedef Kokkos::OpenMP     ExecSpace;
            typedef Kokkos::HostSpace   MemSpace;
            typedef Kokkos::LayoutLeft    Layout;
        #elif defined(ACTIVATE_KOKKOS_OPENMP_RIGHT)
            typedef Kokkos::OpenMP     ExecSpace;
            typedef Kokkos::HostSpace   MemSpace;
            typedef Kokkos::LayoutRight   Layout;
        #elif defined(ACTIVATE_KOKKOS_CUDA_LEFT)
            typedef Kokkos::Cuda       ExecSpace;
            typedef Kokkos::CudaSpace   MemSpace;
            typedef Kokkos::LayoutLeft    Layout;
        #elif defined(ACTIVATE_KOKKOS_CUDA_RIGHT)
            typedef Kokkos::Cuda       ExecSpace;
            typedef Kokkos::CudaSpace   MemSpace;
            typedef Kokkos::LayoutRight   Layout;
        #elif defined(ACTIVATE_SIMULATION)
            typedef Kokkos::Cuda       ExecSpace;
            typedef Kokkos::CudaSpace   MemSpace;
            typedef Kokkos::LayoutLeft    Layout;
            benchmark = false;
        #elif defined(ACTIVATE_GRAPH_KOKKOS_CUDA_LEFT)
            typedef Kokkos::Cuda       ExecSpace;
            typedef Kokkos::CudaSpace   MemSpace;
            typedef Kokkos::LayoutLeft    Layout;
            use_graph = true;
        #elif defined(ACTIVATE_GRAPH_KOKKOS_CUDA_RIGHT)
            typedef Kokkos::Cuda       ExecSpace;
            typedef Kokkos::CudaSpace   MemSpace;
            typedef Kokkos::LayoutRight   Layout;
            use_graph = true;
        #else
            #error ERROR, unable to initialize properly Kokkos environemnt
        #endif

        // Used by for loops
        typedef Kokkos::RangePolicy<ExecSpace> range_policy;

        // Total number of iterations to make simulation
        int nb_iterations = maxt/deltat;

        // Used to store bodies, x and y positions
        typedef Kokkos::View<Body*,            MemSpace>     ViewBodies;
        typedef Kokkos::View<double**, Layout, MemSpace>  ViewPositions;
        typedef Kokkos::View<double**, Layout, MemSpace>      ViewSpeed;
        typedef Kokkos::View<int*,             MemSpace>     ViewTime;

        // Allocation
        ViewBodies    Bodies("Bodies", n_bodies);
        ViewPositions      x( "x",     n_bodies, nb_iterations);
        ViewPositions      y( "y",     n_bodies, nb_iterations);
        ViewSpeed         vx("vx",     n_bodies, nb_iterations);
        ViewSpeed         vy("vy",     n_bodies, nb_iterations);

        // Create host mirrors of device views.
        ViewBodies::HostMirror    h_Bodies = Kokkos::create_mirror_view( Bodies );
        ViewPositions::HostMirror      h_x = Kokkos::create_mirror_view( x );
        ViewPositions::HostMirror      h_y = Kokkos::create_mirror_view( y );
        ViewSpeed::HostMirror         h_vx = Kokkos::create_mirror_view( vx );
        ViewSpeed::HostMirror         h_vy = Kokkos::create_mirror_view( vy );

        // Measuring sequential time (1)
        Kokkos::Timer timer_sequential;

        // Loading data from simulation file
        for(int i = 0 ; i < n_bodies ; i++)
        {
            // Initial positions of particles
            fscanf(sim_data, "%lf", &h_x(i, 0));
            fscanf(sim_data, "%lf", &h_y(i, 0));

            // Initial velocity & Particle mass
            fscanf(sim_data, "%lf", &h_Bodies(i).vx);
            fscanf(sim_data, "%lf", &h_Bodies(i).vy);
            fscanf(sim_data, "%lf", &h_Bodies(i).mass);

            // Initial velocity & Particle mass
            if (benchmark == false)
            {
                h_vx(i, 0) = h_Bodies(i).vx;
                h_vy(i, 0) = h_Bodies(i).vy;
            }

            // Initial force
            h_Bodies(i).fx = 0;
            h_Bodies(i).fy = 0;
        }

        // Closing the simulation file
        fclose(sim_data);

        // Measuring sequential time (2)
        double time_seq = timer_sequential.seconds();

        // Displaying information for python script to take it
        if (benchmark == true)
            std::cout << "Time - Sequential = " << time_seq << "[s]" << std::endl;

        // Deep copy host views to device views.
        Kokkos::deep_copy( Bodies, h_Bodies );
        Kokkos::deep_copy(      x,      h_x );
        Kokkos::deep_copy(      y,      h_y );
        Kokkos::deep_copy(     vy,     h_vx );
        Kokkos::deep_copy(     vx,     h_vy );

        //--------------------------
        //         Kernels
        //--------------------------
        // Iteration index for main loop
        //Kokkos::View<int*, ExecSpace> t("t", 1);
        Kokkos::View<int*, MemSpace> t("t", 1);

        // Kernel K1
        auto lambda_forceLoop = KOKKOS_LAMBDA (int i)
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
                Bodies(i).fx = 0;
                Bodies(i).fy = 0;

                // --- Computing force increment ---
                for(int j = 0 ; j < n_bodies ; j++)
                {
                    if(i != j)
                    {
                        // Distances
                        dx   = x(j, t(0) - 1) - x(i, t(0) - 1);
                        dy   = y(j, t(0) - 1) - y(i, t(0) - 1);
                        dist = sqrt(dx * dx + dy * dy);

                        // Force (1)
                        F = Bodies(j).mass / (dist * dist + EPS * EPS);

                        // Adding force increment
                        Bodies(i).fx += F * dx / dist;
                        Bodies(i).fy += F * dy / dist;
                    }
                }

                // Force (2)
                Bodies(i).fx *= G * Bodies(i).mass;
                Bodies(i).fy *= G * Bodies(i).mass;
            };

        // Kernel K2
        auto lambda_UpdateLoop = KOKKOS_LAMBDA (int i)
            {
                // Update of speed
                Bodies(i).vx += (deltat * Bodies(i).fx) / Bodies(i).mass;
                Bodies(i).vy += (deltat * Bodies(i).fy) / Bodies(i).mass;

                // Adding speed to beautify simulation
                if (benchmark == false)
                {
                    vx(i, t(0)) = Bodies(i).vx;
                    vy(i, t(0)) = Bodies(i).vy;
                }

                // Update of positions
                x(i, t(0)) = x(i, t(0) - 1) + deltat * Bodies(i).vx;
                y(i, t(0)) = y(i, t(0) - 1) + deltat * Bodies(i).vy;
            };

        // Kokkos graph {O -> K1 -> K2}
        auto graph = Kokkos::Experimental::create_graph(
            [=](auto root){
                auto node_level_1 = root.then_parallel_for(        "level1", n_bodies, lambda_forceLoop);
                auto node_level_2 = node_level_1.then_parallel_for("level2", n_bodies, lambda_UpdateLoop);
            }
        );

        //--------------------------
        //       Simulation
        //--------------------------
        // Timer for parallel
        Kokkos::Timer timer_parallel;

        // --- Run the Kokkos graph ---
        for(int t_iter = 1; t_iter < nb_iterations ; t_iter++)
        {
            // Send t_iter to Execution space
            Kokkos::deep_copy(t, t_iter);

            // CASE 1 - Sequential // loops
            if (use_graph == false)
            {
                Kokkos::parallel_for("Bodies - Force Loop",  range_policy(0, n_bodies), lambda_forceLoop);
                Kokkos::parallel_for("Bodies - Update Loop", range_policy(0, n_bodies), lambda_UpdateLoop);
            }

            // CASE 2 - Cuda graphs (removing overhead)
            else
                graph.submit();
        }

        // Stores the parallel time
        double time_par = timer_parallel.seconds();

        // Displaying information for python script to take it
        if (benchmark == true)
        {
            // Kernel 1
            size_t DataSize_K1 = n_bodies * n_bodies * sizeof(double) * 6 + // dx, dy, dist, F, fx, fy
                                 n_bodies * sizeof(double) * 10;            // G, EPS, dx, dist, F, fx, fy, fx', fy'

            // Kernel 2
            size_t DataSize_K2 = n_bodies * sizeof(double) * 4;             // vx, vy, x, y

            // Computing bandwidth
            double Bandwidth = nb_iterations * (DataSize_K1 + DataSize_K2)/time_par;

            // Displaying results
            std::cout << "Time - Parallel   = " << time_par              << "[s]"    << std::endl;
            std::cout << "Time - Total      = " << time_seq + time_par   << "[s]"    << std::endl;
            std::cout << "Bandwidth         = " << Bandwidth/1073741824  << "[GB/s]" << std::endl;
            std::cout << "Number of threads = " << omp_get_max_threads() << "[-]"    << std::endl;
            if (use_graph == false)
                std::cout << "Graphs            = False" << std::endl;
            else
                std::cout << "Graphs            = True" << std::endl;
        }

        if (benchmark == false)
        {
            // Deep copy device views to host views.
            Kokkos::deep_copy(  h_x,  x );
            Kokkos::deep_copy(  h_y,  y );
            Kokkos::deep_copy( h_vx, vx );
            Kokkos::deep_copy( h_vy, vy );

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
                    fprintf(sim_results_x,  "%lf ",  h_x(i, t));
                    fprintf(sim_results_y,  "%lf ",  h_y(i, t));
                    fprintf(sim_results_vx, "%lf ", h_vx(i, t));
                    fprintf(sim_results_vy, "%lf ", h_vy(i, t));
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
        }
    }
    Kokkos::finalize();
    return 0;
}
