#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4


int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            timer.start();
            labwork.labwork5_GPU();
	    printf("labwork 5 GPU ellapsed %.1fms\n", timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
    
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {             // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    //int c = 0;
    int numberOfCores = 0;
    cudaDeviceProp properties;
    cudaGetDeviceCount(&numberOfCores);
    printf("There is %d cores in our machine\n", numberOfCores);
    /*while (properties.name[c] != '\0') {
      printf("%c", properties.name[c]);
      c++;
    }
    printf("\n");*/
    for (int i=0; i<numberOfCores; i++){
        cudaGetDeviceProperties(&properties, i);
        printf("- For the device %d that is called %s \n The clockrate is of %d kHz \n The warp size in threads is of %d\n This device has %d multiprocessors\n The memory clockrate is of %d kHz\n The memory bus width is of %d bits\n", i, properties.name, properties.clockRate, properties.warpSize, properties.multiProcessorCount, properties.memoryClockRate, properties.memoryBusWidth);

    }     
}


//----------------------------------------------------------------------

__global__ void grayscale(uchar3 *input, uchar3 *output) {
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   output[tid].x = (input[tid].x + input[tid].y +
   input[tid].z) / 3;
   output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
   uchar3 *devInput;
   uchar3 * devGray;
   int pixelCount = inputImage->width * inputImage->height;
   int dimBlock = 256;
   int dimGrid = pixelCount/dimBlock;
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   grayscale<<<dimGrid, dimBlock>>>(
   devInput, devGray);
   cudaMemcpy(outputImage, devGray,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devGray);
}


//-----------------------------------------------------------------------

__global__ void grayscale2D(uchar3 *input, uchar3 *output, int width) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   output[tid].x = (input[tid].x + input[tid].y +
   input[tid].z) / 3;
   output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
   uchar3 *devInput;
   uchar3 * devGray;
   int width = inputImage->width;
   int pixelCount = inputImage->width * inputImage->height;
   dim3 gridSize = dim3((inputImage->width)/32, (inputImage->height)/32);
   dim3 blockSize = dim3(32, 32);
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   grayscale2D<<<gridSize, blockSize>>>(
   devInput, devGray, width);
   cudaMemcpy(outputImage, devGray,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devGray);   
}


//------------------------------------------------------------------------

__global__ void gauss_unshared(uchar3 *input, uchar3 *output, int height, int width) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
   if (tidx>=width){return;}
   if (tidy>=height){return;}
   int sum = 0;
   int c = 0;
   for (int y = -3; y <= 3; y++) {
       for (int x = -3; x <= 3; x++) {
           int i = tidx + x;
           int j = tidy + y;
           if (i < 0) continue;
           if (i >= width) continue;
           if (j < 0) continue;
           if (j >= height) continue; 
           int tid = j * width + i;
           unsigned char gray = (input[tid].x + input[tid].y + input[tid].z)/3;
           int coefficient = kernel[(y+3) * 7 + x + 3];
           sum = sum + gray * coefficient;
           c += coefficient;
      }
  }
  sum /= c;
  int posOut = tidy * width + tidx;
  output[posOut].x = output[posOut].y = output[posOut].z = sum;
   
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

void Labwork::labwork5_GPU() {
   uchar3 *devInput;
   uchar3 * devGauss;
   int pixelCount = inputImage->width * inputImage->height;
   dim3 gridSize = dim3((inputImage->width)/32, (inputImage->height)/32);
   dim3 blockSize = dim3(32, 32);
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devGauss, pixelCount * sizeof(uchar3));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   gauss_unshared<<<gridSize, blockSize>>>(
   devInput, devGauss, inputImage->height, inputImage->width);
   cudaMemcpy(outputImage, devGauss,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devGauss);   
    
}


//------------------------------------------------------------------



__global__ void brightness(uchar3 *input, uchar3 *output, int width, int increase) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   int newValuex = input[tid].x + increase;
   int newValuey = input[tid].y + increase;
   int newValuez = input[tid].z + increase;

   if (newValuex>0){
      output[tid].x = min(newValuex, 255);
   } else {
      output[tid].x = 0;
   }

   if (newValuey>0){
      output[tid].y = min(newValuey, 255);
   } else {
      output[tid].y = 0;
   }

   if (newValuez>0){
      output[tid].z = min(newValuez, 255);
   } else {
      output[tid].z = 0;
   }
}

__global__ void binarization(uchar3 *input, uchar3 *output, int width, int t) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   if (((input[tid].x + input[tid].y + input[tid].z )/3) < t){
      output[tid].x = output[tid].y = output[tid].z = 0;
   } else {
      output[tid].x = output[tid].y = output[tid].z = 255;
   }
}

__global__ void blending(uchar3 *input1, uchar3 *input2, uchar3 *output, int width, int c) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   output[tid].x = output[tid].y = output[tid].z = c*input1[tid].x + (1-c)*input2[tid].x;
}


void Labwork::labwork6_GPU() {
   uchar3 *devInput;
   uchar3 * devMap;
   int t = 150;
   int pixelCount = inputImage->width * inputImage->height;
   dim3 gridSize = dim3((inputImage->width)/32, (inputImage->height)/32);
   dim3 blockSize = dim3(32, 32);
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devMap, pixelCount * sizeof(uchar3));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   binarization<<<gridSize, blockSize>>>(
   devInput, devMap, inputImage->width, t);
   cudaMemcpy(outputImage, devMap,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devMap);   

}

//------------------------------------------------------------------

extern __shared__ int cache[];

__global__ void reduceFinalMin(uchar3 *in, int *out, int width) {
   // dynamic shared memory size, allocated in host
   // cache the block content
   unsigned int localtid = threadIdx.x;
   int tidx = threadIdx.x + blockIdx.x* 2 * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y* 2 * blockDim.y;
   int tid = tidy * width + tidx;
   //unsigned int tid = threadIdx.x+blockIdx.x*2*blockDim.x;
   if ( in[tid].x < in[tid + blockDim.x].x ) {
      cache[localtid] = in[tid].x;
   } else {
      cache[localtid] = in[tid + blockDim.x].x;
   }
   __syncthreads();
   // reduction in cache
   for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (localtid < s) {
         if (cache[localtid] > cache[localtid + s]){
            cache[localtid] = cache[localtid + s];
         } //else : cache[localtid] = cache[localtid], no need to code it
      }
      __syncthreads();
   }
   // only first thread writes back
   if (localtid == 0) {*out = cache[0];}
}


__global__ void reduceFinalMax(uchar3 *in, int *out, int width) {
   // dynamic shared memory size, allocated in host

   // cache the block content
   unsigned int localtid = threadIdx.x;
   int tidx = threadIdx.x + blockIdx.x* 2 * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y* 2 * blockDim.y;
   int tid = tidy * width + tidx;
   //unsigned int tid = threadIdx.x+blockIdx.x*2*blockDim.x;
   if ( in[tid].x > in[tid + blockDim.x].x ) {
      cache[localtid] = in[tid].x;
   } else {
      cache[localtid] = in[tid + blockDim.x].x;
   }
   __syncthreads();
   // reduction in cache
   for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (localtid < s) {
         if (cache[localtid] < cache[localtid + s]){
            cache[localtid] = cache[localtid + s];
         } //else : cache[localtid] = cache[localtid], no need to code it
      }
      __syncthreads();
   }
   // only first thread writes back
   if (localtid == 0) {*out = cache[0];}
}

__global__ void grayscaleStretch(uchar3 *input, uchar3 *output, int min, int max, int width) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   output[tid].x = ((input[tid].x - min) / (max-min))*255 ; 
//We know our picture is in gray scale, so x=y=z, we just have to use x.
   output[tid].z = output[tid].y = output[tid].x;
}


void Labwork::labwork7_GPU() {
   uchar3 *devInput;
   uchar3 * devGrayStretch;
   int min;
   int max;
   int pixelCount = inputImage->width * inputImage->height;
   int width = inputImage->width;
   dim3 gridSize = dim3((inputImage->width)/16, (inputImage->height)/16);
   dim3 blockSize = dim3(16, 16);
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devGrayStretch, pixelCount * sizeof(uchar3));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   grayscale2D<<<gridSize, blockSize>>>(
   devInput, devGrayStretch, width);
   reduceFinalMax<<<gridSize, blockSize>>>(
   devInput, &max, width);
   reduceFinalMin<<<gridSize, blockSize>>>(
   devInput, &min, width);
   grayscaleStretch<<<gridSize, blockSize>>>(
   devInput, devGrayStretch, min, max, width);
   cudaMemcpy(outputImage, devGrayStretch,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devGrayStretch);   

}


//--------------------------------------------------------------------

__global__ void RGB2HSV(uchar3 *input, float *outH, float *outS, float *outV,int width){
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   int max;
   int min;
   int delta;
   float r = input[tid].x / 255;
   float g = input[tid].y / 255;
   float b = input[tid].z / 255;
   if (r>g && r>b){
      max = r;
      if (b>g){
         min = g;
      } else {
         min = b;
      }
   } else if (g>b){
      max = g;
      if (b>r){
         min = r;
      } else {
         min = b;
      }
   } else {
      max = b;
      if (g>r){
         min = r;
      } else {
         min = g;
      }
   }

   delta = max - min;

   outV[tid] = max;

   if (delta == 0){
      outH[tid] = 0;
   } else {
      if (max == r){
         outH[tid] = float(60 * (int((g-b)/delta) % 6));
      } else if (max == g){
         outH[tid] = 60 * (((b-r)/delta) + 2);
      } else if (max == b){
         outH[tid] = 60 * (((r-g)/delta) + 4);
      }
   }
   if (max == 0){
      outS[tid] = 0;
   } else {
      outS[tid] = delta/max;
   }
}

__global__ void HSV2RGB(uchar3 *output, float *inH, float *inS, float *inV, int width){
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   float h = inH[tid];
   float s = inS[tid];
   float v = inV[tid];
   float d = h/60;
   float l = v * (1 - s);
   int hi = int(d) % 6;
   float f = d - hi;
   float m = v * (1 - f * s);
   float n = v * (1 - (1 - f) * s);
   if (h<60 && h>=0){
      output[tid].x = v;
      output[tid].y = n;
      output[tid].z = l;
   } else if (h<120 && h>=60){
      output[tid].x = m;
      output[tid].y = v;
      output[tid].z = l;
   } else if (h<180 && h>=120){
      output[tid].x = l;
      output[tid].y = v;
      output[tid].z = n;
   } else if (h<240 && h>=180){
      output[tid].x = l;
      output[tid].y = m;
      output[tid].z = v;
   } else if (h<300 && h>=240){
      output[tid].x = n;
      output[tid].y = l;
      output[tid].z = v;
   } else if (h<360 && h>=300){
      output[tid].x = v;
      output[tid].y = l;
      output[tid].z = m;
   }
   output[tid].x = int(output[tid].x*255);
   output[tid].y = int(output[tid].y*255);
   output[tid].z = int(output[tid].z*255);
}

void Labwork::labwork8_GPU() {
   uchar3 *devInput;
   uchar3 * devOutput;
   float * devH;
   float * devS;
   float * devV;
   int pixelCount = inputImage->width * inputImage->height;
   dim3 gridSize = dim3((inputImage->width)/32, (inputImage->height)/32);
   dim3 blockSize = dim3(32, 32);
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devH, pixelCount * sizeof(float));
   cudaMalloc(&devS, pixelCount * sizeof(float));
   cudaMalloc(&devV, pixelCount * sizeof(float));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   RGB2HSV<<<gridSize, blockSize>>>(
   devInput, devH, devS, devV, inputImage->width);
   HSV2RGB<<<gridSize, blockSize>>>(
   devOutput, devH, devS, devV, inputImage->width);
   cudaMemcpy(outputImage, devOutput,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devOutput);  
}

//----------------------------------------------------------------------------

__global__ void histogramMaker(uchar3 *input, int *values, int width) {
   int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   int tidy = threadIdx.y + blockIdx.y * blockDim.y;
   int tid = tidy * width + tidx;
   values[input[tid].x]+=1; //We have a gray image so x=y=z
}

__global__ void equalization(int *hist, int *eqHist, int n) {
   int c[255];
   int p[255];
   for (int i=0; i<255; i++){
      p[i] = hist[i]/n;
   }
   for (int i=0; i<255; i++){
      for (int j=0; j<=i; j++){
          c[i]+=p[j];
      }
      eqHist[i] = c[i]*255;
   }
}


void Labwork::labwork9_GPU() {
   uchar3 *devInput;
   uchar3 * devGray;
   uchar3 * devOutput;
   int hist[255];
   int eqHist[255];
   int pixelCount = inputImage->width * inputImage->height;
   int width = inputImage->width;
   dim3 gridSize = dim3((inputImage->width)/32, (inputImage->height)/32);
   dim3 blockSize = dim3(32, 32);
   outputImage = static_cast<char *>(malloc(pixelCount * 3));
   cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
   cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
   cudaMalloc(&devOutput, pixelCount * sizeof(uchar3));
   cudaMemcpy(devInput, inputImage->buffer,
   pixelCount * sizeof(uchar3),
   cudaMemcpyHostToDevice);
   grayscale2D<<<gridSize, blockSize>>>(
   devInput, devGray, width);
   histogramMaker<<<gridSize, blockSize>>>(
   devGray, hist, width);
   for (int i=0; i<255; i++){
      printf("%d pixel sont %d |",hist[i],i);
   }
   equalization<<<gridSize, blockSize>>>(hist, eqHist, pixelCount);
   printf("\n\n\n\n\n");
   for (int i=0; i<255; i++){
      printf("%d pixel sont %d |",eqHist[i],i);
   }
   cudaMemcpy(outputImage, devOutput,
   pixelCount * sizeof(uchar3),
   cudaMemcpyDeviceToHost);
   cudaFree(devInput);
   cudaFree(devGray);  
   cudaFree(devOutput); 
}

void Labwork::labwork10_GPU() {

}
