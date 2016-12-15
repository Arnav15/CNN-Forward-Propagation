#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

#define KERNEL_WIDTH 5

#define BLOCK_SIZE 64
#define TILE_WIDTH 12
#define TILE_DIM 16
#define POOL_SIZE 2


static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}
__global__ void convolution(int in_width, int out_width, int in_channel, int out_channel, 
                            int W_grid, float* X, float* W, float* Y) 
{
    int i, j, n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + KERNEL_WIDTH-1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    h = h_base+ h0;
    w = w_base+ w0;
    float acc = 0.0;

    for (int c = 0; c < in_channel; c++) 
    {
        if (( h0 < KERNEL_WIDTH) && ( w0 < KERNEL_WIDTH))
             W_shared[h0*KERNEL_WIDTH+ w0]= W[ (h0 * KERNEL_WIDTH * in_channel* out_channel) + (w0* in_channel*out_channel) +(c*out_channel)+m]; 
         __syncthreads();

        for (i = h; i < h_base+ X_tile_width; i += TILE_WIDTH) 
        {
            for (j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
            {
                if(i < in_width && j < in_width)
                  X_shared[(i-h_base)*X_tile_width+ (j-w_base)] = X[(n*in_width*in_width*in_channel)+(i*in_width*in_channel)+(j*in_channel)+c]; 
            }
        } 
        __syncthreads();

        for (i = 0; i < KERNEL_WIDTH; i++) 
        {
            for (j = 0; j < KERNEL_WIDTH; j++) 
            { 
                if(h < out_width && w < out_width)
                    acc = acc + X_shared[(h0 + i)*X_tile_width + (w0+j)] * W_shared[i *KERNEL_WIDTH + j];
            }
        }
        __syncthreads();
    }
    if(h < out_width && w < out_width)
    {
        int Yoffset = ((n * out_width + h) * out_width + w) * out_channel + m;
        Y[Yoffset] = (int)(acc > 0) * acc;
    }
}

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float subTileM[TILE_DIM][TILE_DIM];
    __shared__ float subTileN[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + numAColumns - 1)/TILE_DIM; i++) {

         if (i*TILE_DIM + threadIdx.x < numAColumns && Row < numARows)   
            subTileM[threadIdx.y][threadIdx.x] = A[Row*numAColumns + i*TILE_DIM + threadIdx.x];
         else                                                   
            subTileM[threadIdx.y][threadIdx.x] = 0.0;

         if (i*TILE_DIM + threadIdx.y < numBRows && Col < numBColumns)   
            subTileN[threadIdx.y][threadIdx.x] = B[(i*TILE_DIM + threadIdx.y)*numBColumns + Col];
         else                                                   
            subTileN[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         if(Row<numARows && Col < numBColumns)
         {
            for (int j = 0; j < TILE_DIM; j++) 
                CValue += subTileM[threadIdx.y][j] * subTileN[j][threadIdx.x];
          }

         __syncthreads();
    }

    if (Row < numCRows && Col < numCColumns) 
        C[((blockIdx.y * blockDim.y + threadIdx.y)*numCColumns)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}


__global__ void pooling(int in_width, int out_width, int in_channel, int out_channel, float* X, float* Y) 
{
  int bx,by,tx,ty;
  bx = blockIdx.x; 
  by = blockIdx.y;
  tx = threadIdx.x;
  ty = threadIdx.y;
  float acc = 0.0;
  int Yoffset = (bx * out_width * out_width * out_channel) + (ty * out_width * out_channel) + (tx * out_channel) + by;
  int pool_s = 2;

  for (int p = 0; p < POOL_SIZE; p++) 
  {
    for (int q = 0; q < POOL_SIZE; q++) 
      acc += X[(bx * in_width * in_width * in_channel) + (((POOL_SIZE * ty) + p) * in_width * in_channel)
             + ((POOL_SIZE * tx + q) * in_channel) + by]/(1.0f * pool_s * pool_s);
                           
  }
  Y[Yoffset] = acc;
}

// From book chapter Figure 16.4
void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  // const auto filter_h   = wdims[0];
  // const auto filter_w   = wdims[1];
  // const auto in_channel = wdims[2];

  // for (const auto i : range(0, ydims[0])) {
  //   for (const auto m : range(0, ydims[3])) {
  //     for (const auto w : range(0, ydims[2])) {
  //       for (const auto h : range(0, ydims[1])) {
  //         for (const auto p : range(0, filter_h)) {
  //           for (const auto q : range(0, filter_w)) {
  //             for (const auto c : range(0, in_channel)) {
  //               const auto yoffset =
  //                   ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
  //               const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
  //                                    (h + p) * xdims[2] * xdims[3] +
  //                                    (w + q) * xdims[3] + c;
  //               const auto woffset = p * wdims[1] * wdims[2] * wdims[3] +
  //                                    q * wdims[2] * wdims[3] + c * wdims[3] + m;
  //               Y[yoffset] += X[xoffset] * W[woffset];
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

    float* d_input_img;
    float* convout;
    cudaMalloc((void **) &d_input_img, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
    cudaMalloc((void **) &convout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3]);
    cudaMemcpy(d_input_img, X, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3], cudaMemcpyHostToDevice);

    int W_grid = ceil(ydims[1]/(float)TILE_WIDTH);
    int H_grid = ceil(ydims[1]/(float)TILE_WIDTH);
    int Z = H_grid * W_grid;

    dim3 dimGrid(xdims[0],ydims[3],Z);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH);

    size_t shmem_size = sizeof(float) * ( (TILE_WIDTH + KERNEL_WIDTH-1)*(TILE_WIDTH + KERNEL_WIDTH-1) + KERNEL_WIDTH*KERNEL_WIDTH );

    float* filter_conv;
    cudaMalloc((void **) &filter_conv, sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3]);
    cudaMemcpy(filter_conv, W, sizeof(float) * conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3], cudaMemcpyHostToDevice);
    convolution<<<dimGrid, dimBlock, shmem_size>>>(xdims[1], ydims[1], xdims[3], ydims[3],
                                                   2, d_input_img, filter_conv, convout);
    cudaFree(filter_conv);

    cudaMemcpy(Y, convout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3], cudaMemcpyDeviceToHost);
    cudaFree(convout);
    cudaFree(d_input_img);
    return;
}


// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}


// From book chapter Figure 16.5
void average_pool(const float *X, const int xdims[4],
                         float *Y, const int ydims[4]) {
  // for (const auto i : range(0, ydims[0])) {
  //   for (const auto m : range(0, ydims[3])) {
  //     for (const auto w : range(0, ydims[2])) {
  //       for (const auto h : range(0, ydims[1])) {
  //         for (const auto p : range(0, pool_size)) {
  //           for (const auto q : range(0, pool_size)) {
  //             const auto yoffset =
  //                 ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
  //             const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
  //                                  (pool_size * h + p) * xdims[2] * xdims[3] +
  //                                  (pool_size * w + q) * xdims[3] + m;
  //             Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  float* d_input_img;
  float* poolout;
  cudaMalloc((void **) &d_input_img, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
  cudaMalloc((void **) &poolout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3]);
  cudaMemcpy(d_input_img, X, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3], cudaMemcpyHostToDevice);

  dim3 dimGrid(xdims[0], ydims[3], 1);
  dim3 dimBlock(ydims[1], ydims[1]);
  pooling<<<dimGrid, dimBlock>>>(xdims[1], ydims[1], xdims[3], ydims[3], d_input_img, poolout);

  cudaMemcpy(Y, poolout, sizeof(float) * ydims[0] * ydims[1] * ydims[2] * ydims[3], cudaMemcpyDeviceToHost);
  cudaFree(poolout);
  cudaFree(d_input_img);
  return;

}

void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) {
  // for (const auto i : range(0, xdims[0])) {
  //   for (const auto j : range(0, wdims[1])) {
  //     float sum = 0;
  //     for (const auto k : range(0, xdims[1])) {
  //       sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
  //     }
  //     Y[i * wdims[1] + j] = sum;
  //   }
  // }

  float* d_input_img;
  float* fc1out;
  float* w_matrix;
  cudaMalloc((void **) &d_input_img, sizeof(float) * xdims[0] * xdims[1]);
  cudaMalloc((void **) &fc1out, sizeof(float) * ydims[0] * ydims[1]);
  cudaMalloc((void **) &w_matrix, sizeof(float)*wdims[0] * wdims[1]);

  cudaMemcpy(d_input_img, X, sizeof(float) * xdims[0] * xdims[1], cudaMemcpyHostToDevice);
  cudaMemcpy(w_matrix, W, sizeof(float) * wdims[0]*wdims[1], cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(ydims[1]/(float)TILE_DIM), ceil(ydims[0]/(float)TILE_DIM));
  dim3 dimBlock(TILE_DIM, TILE_DIM);

  matrixMultiplyShared<<<dimGrid,dimBlock>>>(d_input_img, w_matrix, fc1out,
                                            xdims[0], xdims[1],
                                            wdims[0], wdims[1],
                                            ydims[0], ydims[1]);
  cudaMemcpy(Y, fc1out, sizeof(float) * ydims[0] * ydims[1], cudaMemcpyDeviceToHost);

  cudaFree(d_input_img);
  cudaFree(fc1out);
  cudaFree(w_matrix);

}

// Choose the guess with largest score
void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  average_pool(a, adims, b, bdims);

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  // relu
  relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  average_pool(c, cdims, d, ddims);

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}

