#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

#define TILE 16

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static vector3 *d_pos   = NULL;
static vector3 *d_vel   = NULL;
static vector3 *d_accels = NULL;
static double  *d_mass  = NULL;
static int      initialized = 0;

//pairwise_kernel: GPU kernel that computes the pairwise accelerations between all entities.
//                 Replaces the first nested for-loop in the serial compute().
//                 Effect is on the first argument (row = affected body).
//Parameters: pos    - device array of positions
//            mass   - device array of masses
//            accels - device NxN acceleration matrix (flat)
//Returns: None
//Side Effect: Fills accels[i*NUMENTITIES+j] with the acceleration that body j exerts on body i.
//             Uses shared memory tiling (TILE x 3 positions + TILE masses) to reduce
//             global memory traffic -- each tile of source bodies is loaded once and
//             reused by all TILE rows of threads in the block.
__global__ void pairwise_kernel(vector3 *pos, double *mass, vector3 *accels)
{
    //shared memory tiles for the current strip of source bodies (j dimension)
    __shared__ double shPos[TILE][3];
    __shared__ double shMass[TILE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NUMENTITIES) return;

    //cache body i's position in registers -- read once from global memory
    double myPos[3] = {pos[i][0], pos[i][1], pos[i][2]};

    //tile loop: load source bodies TILE at a time into shared memory
    //           then compute interactions against that cached tile
    for (int tile = 0; tile < (NUMENTITIES + TILE - 1) / TILE; tile++)
    {
        int j_global = tile * TILE + threadIdx.x;

        //first row of threads in the block cooperatively loads the tile
        if (threadIdx.y == 0 && j_global < NUMENTITIES) {
            shPos[threadIdx.x][0] = pos[j_global][0];
            shPos[threadIdx.x][1] = pos[j_global][1];
            shPos[threadIdx.x][2] = pos[j_global][2];
            shMass[threadIdx.x]   = mass[j_global];
        }

        __syncthreads();  //wait for tile to be fully loaded before computing

        //first compute the pairwise accelerations.  Effect is on the first argument.
        if (j_global < NUMENTITIES) {
            int index = i * NUMENTITIES + j_global;

            if (i == j_global) {
                //diagonal: no self-interaction
                accels[index][0] = 0;
                accels[index][1] = 0;
                accels[index][2] = 0;
            } else {
                //compute distance vector from i to j (matches serial: distance[k]=hPos[i][k]-hPos[j][k])
                double distance[3];
                distance[0] = myPos[0] - shPos[threadIdx.x][0];
                distance[1] = myPos[1] - shPos[threadIdx.x][1];
                distance[2] = myPos[2] - shPos[threadIdx.x][2];

                double magnitude_sq = distance[0]*distance[0]
                                    + distance[1]*distance[1]
                                    + distance[2]*distance[2];
                double magnitude  = sqrt(magnitude_sq);
                double accelmag   = -1 * GRAV_CONSTANT * shMass[threadIdx.x] / magnitude_sq;

                accels[index][0] = accelmag * distance[0] / magnitude;
                accels[index][1] = accelmag * distance[1] / magnitude;
                accels[index][2] = accelmag * distance[2] / magnitude;
            }
        }

        __syncthreads();  //wait before overwriting shared memory in next tile iteration
    }
}

//sum_kernel: GPU kernel that sums up the rows of the acceleration matrix to get
//            the net effect on each entity, then updates velocity and position.
//            Replaces the second nested for-loop in the serial compute().
//Parameters: pos    - device array of positions
//            vel    - device array of velocities
//            accels - device NxN acceleration matrix (flat)
//Returns: None
//Side Effect: Updates pos and vel arrays in device memory for one INTERVAL.
__global__ void sum_kernel(vector3 *pos, vector3 *vel, vector3 *accels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUMENTITIES) return;

    //sum up the rows of our matrix to get effect on each entity
    vector3 accel_sum = {0, 0, 0};
    for (int j = 0; j < NUMENTITIES; j++) {
        int idx = i * NUMENTITIES + j;
        accel_sum[0] += accels[idx][0];
        accel_sum[1] += accels[idx][1];
        accel_sum[2] += accels[idx][2];
    }

    //compute the new velocity based on the acceleration and time interval
    //compute the new position based on the velocity and time interval
    for (int k = 0; k < 3; k++) {
        vel[i][k] += accel_sum[k] * INTERVAL;
        pos[i][k] += vel[i][k] * INTERVAL;
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute()
{
    //make an acceleration matrix which is NUMENTITIES squared in size;
    //allocate device memory once on first call and reuse every subsequent step
    if (!initialized) {
        size_t vecSize   = sizeof(vector3) * NUMENTITIES;
        size_t massSize  = sizeof(double)  * NUMENTITIES;
        size_t accelSize = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

        CUDA_CHECK(cudaMalloc(&d_pos,    vecSize));
        CUDA_CHECK(cudaMalloc(&d_vel,    vecSize));
        CUDA_CHECK(cudaMalloc(&d_mass,   massSize));
        CUDA_CHECK(cudaMalloc(&d_accels, accelSize));

        CUDA_CHECK(cudaMemcpy(d_pos,  hPos,  vecSize,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel,  hVel,  vecSize,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, mass,  massSize, cudaMemcpyHostToDevice));

        initialized = 1;
    }

    //first compute the pairwise accelerations -- 2D grid covers the NxN matrix
    dim3 threads2D(TILE, TILE);
    dim3 blocks2D((NUMENTITIES + TILE - 1) / TILE,
                  (NUMENTITIES + TILE - 1) / TILE);

    pairwise_kernel<<<blocks2D, threads2D>>>(d_pos, d_mass, d_accels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //sum up the rows of our matrix to get effect on each entity, then update velocity and position
    int threads1D = 256;
    int blocks1D  = (NUMENTITIES + threads1D - 1) / threads1D;

    sum_kernel<<<blocks1D, threads1D>>>(d_pos, d_vel, d_accels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //copy updated positions and velocities back to host so printSystem() works
    CUDA_CHECK(cudaMemcpy(hPos, d_pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel, d_vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost));
}

//compute_cleanup: Frees device memory allocated by the first call to compute().
//Parameters: None
//Returns: None
//Side Effects: Frees d_pos, d_vel, d_mass, and d_accels from device memory.
void compute_cleanup()
{
    if (initialized) {
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_mass));
        CUDA_CHECK(cudaFree(d_accels));
        initialized = 0;
    }
}
