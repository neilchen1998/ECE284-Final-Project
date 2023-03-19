#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <string>
#include <sstream>
#include <deque>

#include "timer.h"
#include "utils.h"

#define MAX_SEQ_LEN 1024
#define MAX_BLOCK_SEQ_LEN 64

// the paramaters for the pairwiseHMM
#define DELTA 0.3
#define EPSILON 0.8
#define MATCH 0.99
#define MISMATCH 0.01
#define GAP 0.25

// a macro that returns the largest value from two inputs
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// the toggle for printing our many pieces of info (only when debugging)
#define DEBUG 0

/// @brief the kernel for the parallel kernel
/// @param ref the pointer that points to the beginning of the ref. sequence
/// @param query the pointer that points to the beginning of the ref. sequence
/// @param refLen the length of the ref. sequences
/// @param queryLen the length of the query sequences
/// @param VP the pointer that points to the beginning of the global memory that stores the pointers
/// @return 
__global__ void parallel(char* ref, char* query, int refLen, int queryLen, int* VP)
{
    // the share memory
    extern __shared__ float array[];

    // the index
    int idx = threadIdx.x;
    int bx = blockIdx.x;
    
    // calculate the ref len
    int newRefLen = 0;
    if ((refLen - bx*MAX_BLOCK_SEQ_LEN) >= MAX_BLOCK_SEQ_LEN)
        newRefLen = MAX_BLOCK_SEQ_LEN;
    else 
        newRefLen = (refLen - bx*MAX_BLOCK_SEQ_LEN);

    // divide the share memory into three segments
    float* VM = (float*)array;
    float* VI = (float*)&VM[newRefLen * newRefLen];
    float* VJ = (float*)&VI[newRefLen * newRefLen];
    float currM = 0.0;
    float currI = 0.0;
    float currJ = 0.0;

    // calculate the offset value
    int offset = bx*MAX_BLOCK_SEQ_LEN;

    // initialize matrices
    for (std::size_t i = threadIdx.x; i < newRefLen * newRefLen; i += blockDim.x)
    {
        *(VM+i) = 0.0001;
        *(VI+i) = 0.0001;
        *(VJ+i) = 0.0001;
    }

    if (threadIdx.x == 0)
    {
        // start with a match case
        if (*(ref+offset) == *(query+offset))
            *(VM) = 1.0;
        // start with a delete case
        else if (*(ref + 1+offset) == *(query+offset))
            *(VJ) = 1.0;
        // start with an insert case
        else 
            *(VI) = 1.0;
    } 

    #if DEBUG

        if (threadIdx.x == 0 && blockIdx.x == 1)
        {
            printf("*** Block %d ***\n", blockIdx.x);
            printf("*** Block VM ***\n");
            for (size_t i = 0; i < 8; i++)
            {
                for (size_t j = 0; j < 8; j++)
                {
                    printf("%.2f ", *(VM+8*i+j));
                }
                printf("\n");
            }
            printf("*** Block VI ***\n");
            for (size_t i = 0; i < 8; i++)
            {
                for (size_t j = 0; j < 8; j++)
                {
                    printf("%.2f ", *(VI+8*i+j));
                }
                printf("\n");
            }
            printf("*** Block VJ ***\n");
            for (size_t i = 0; i < 8; i++)
            {
                for (size_t j = 0; j < 8; j++)
                {
                    printf("%.2f ", *(VJ+8*i+j));
                }
                printf("\n");
            }
            printf("*** Block %d End ***\n", blockIdx.x);
        }

    #endif

    __syncthreads();

    // upper triangle
    for (std::size_t x = 0; x < newRefLen; x++)
    {
        if (idx <= x)
        {
            // index is (y, x-y)
            std::size_t y = idx;
            int i = y, j = x - y;

            if (i == 0 && j == 0) continue;

            float a, b, c;

            // VM
            if (i > 0 && j > 0)
            {
                a = (1-2*DELTA) * *(VM+newRefLen*(i-1)+(j-1));
                b = (1-EPSILON) * *(VI+newRefLen*(i-1)+(j-1));
                c = (1-EPSILON) * *(VJ+newRefLen*(i-1)+(j-1));

                // check if they are a match or a mismatch
                if (*(ref+i+offset) == *(query+j+offset))
                    currM  = MATCH * MAX(MAX(a, b), c);
                else
                    currM = MISMATCH * MAX(MAX(a, b), c);

                *(VM+newRefLen*i+j) = currM;
            }

            // VI
            if (j > 0)
            {
                a = DELTA * *(VM+newRefLen*i+(j-1));
                b = EPSILON * *(VI+newRefLen*i+(j-1));
                currI = GAP * MAX(a, b);
                *(VI+newRefLen*i+j) = currI;
            }

            // VD
            if (i > 0)
            {
                a = DELTA * *(VM+newRefLen*(i-1)+j);
                b = EPSILON * *(VJ+newRefLen*i+(j-1));
                currJ = GAP * MAX(a, b);
                *(VJ+newRefLen*i+j) = currJ;
            }

            // VP
            float M = currM, I = currI, J = currJ;
            if (M >= I && M >= J)
                *(VP+newRefLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 0;
            else if (I > J)
                *(VP+newRefLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 2;
            else
                *(VP+newRefLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 1;
        }
        __syncthreads();
    }

    // lower triangle
    for (std::size_t x = newRefLen; x < 2*newRefLen-1; x++)
    {
        if ( idx <= ((2*newRefLen -2) - x))
        {
            // index is (x-N+1+y, N-1-y)
            std::size_t y = idx;
            int i = x - newRefLen + 1 + y, j = newRefLen - 1 - y;

            float a, b, c;

            // VM
            if (i > 0 && j > 0)
            {
                a = (1-2*DELTA) * *(VM+newRefLen*(i-1)+(j-1));
                b = (1-EPSILON) * *(VI+newRefLen*(i-1)+(j-1));
                c = (1-EPSILON) * *(VJ+newRefLen*(i-1)+(j-1));

                if (*(ref+i+offset) == *(query+j+offset))
                    currM  = MATCH * MAX(MAX(a, b), c);
                else
                    currM = MISMATCH * MAX(MAX(a, b), c);

                *(VM+newRefLen*i+j) = currM;
            }


            // VI
            if (j > 0)
            {
                a = DELTA * *(VM+newRefLen*i+(j-1));
                b = EPSILON * *(VJ+newRefLen*i+(j-1));
                currI = GAP * MAX(a, b);

                *(VI+newRefLen*i+j) = currI;
            }

            // VD
            if (i > 0)
            {
                a = DELTA * *(VM+newRefLen*(i-1)+j);
                b = EPSILON * *(VI+newRefLen*(i-1)+j);
                currJ = GAP * MAX(a, b);

                *(VJ+newRefLen*i+j) = currJ;
            }

            // VP
            float M = currM, I = currI, J = currJ;
            if (M >= I && M >= J)
                *(VP+newRefLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 0;
            else if (I > J)
                *(VP+newRefLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 2;
            else
                *(VP+newRefLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 1;
        }
        __syncthreads();
    }
}

/// @brief the kernel for the viterbi kernel (the sequential kernel)
/// @param ref the pointer that points to the beginning of the ref. sequence
/// @param query the pointer that points to the beginning of the ref. sequence
/// @param refLen the length of our two sequences
/// @param VP the pointer that points to the beginning of the global memory that stores the pointers
/// @return 
__global__ void viterbi(char* ref, char* query, int refLen, int* VP)
{
    extern __shared__ float array[];
    float* VM = (float*)array;
    float* VI = (float*)&VM[MAX_BLOCK_SEQ_LEN * MAX_BLOCK_SEQ_LEN];
    float* VJ = (float*)&VI[MAX_BLOCK_SEQ_LEN * MAX_BLOCK_SEQ_LEN];
    int offset = 0;
    std::size_t numIts = 0;
    if ((refLen % MAX_BLOCK_SEQ_LEN) == 0){
        numIts = refLen / MAX_BLOCK_SEQ_LEN;
    } else {
        numIts = (refLen / MAX_BLOCK_SEQ_LEN) + 1;
    }
    int fullLen = refLen;

    if (threadIdx.x == 0)
    {
        for (std::size_t k = 0; k < numIts; k++)
        {
            offset = k * MAX_BLOCK_SEQ_LEN;

            if ((fullLen - k*MAX_BLOCK_SEQ_LEN) >= MAX_BLOCK_SEQ_LEN)
            {
                refLen = MAX_BLOCK_SEQ_LEN;
            } 
            else
            {
                refLen = (fullLen - k*MAX_BLOCK_SEQ_LEN);
            }

            // initialize matrices
            for (std::size_t i = 0; i < refLen * refLen; ++i)
            {
                *(VM+i) = 0.0001;
                *(VI+i) = 0.0001;
                *(VJ+i) = 0.0001;
            }
            
            if (*(ref+offset) == *(query+offset))
            {            
                // start with match case
                *(VM) = 1.0;
            } 
            else if (*(ref + 1+offset) == *(query+offset))
            { 
                // start with delete case
                *(VJ) = 1.0;
            } 
            else
            {                            
                // default case
                *(VI) = 1.0;
            }
        
            float currM, currI, currJ;

            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    if (i == 0 && j == 0) continue;
                    float a, b, c;

                    // VM
                    if (i > 0 && j > 0)
                    {
                        a = (1-2*DELTA) * *(VM+refLen*(i-1)+(j-1));
                        b = (1-EPSILON) * *(VI+refLen*(i-1)+(j-1));
                        c = (1-EPSILON) * *(VJ+refLen*(i-1)+(j-1));

                        if (*(ref+i+offset) == *(query+j+offset))
                            currM  = MATCH * MAX(MAX(a, b), c);
                        else
                            currM = MISMATCH * MAX(MAX(a, b), c);

                        *(VM+refLen*i+j) = currM;
                    }

                    // VI
                    if (j > 0)
                    {
                        a = DELTA * *(VM+refLen*i+(j-1));
                        b = EPSILON * *(VI+refLen*i+(j-1));
                        currI = GAP * MAX(a, b);
                        *(VI+refLen*i+j) = currI;
                    }

                    // VD
                    if (i > 0)
                    {
                        a = DELTA * *(VM+refLen*(i-1)+j);
                        b = EPSILON * *(VJ+refLen*i+(j-1));
                        currJ = GAP * MAX(a, b);
                        *(VJ+refLen*i+j) = currJ;
                    }

                    // VP
                    float M = currM, I = currI, J = currJ;
                    if (M >= I && M >= J)
                        *(VP+refLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 0;
                    else if (I > J)
                        *(VP+refLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 2;
                    else
                        *(VP+refLen*i+j+offset*MAX_BLOCK_SEQ_LEN) = 1;
                }
            }
            __syncthreads();
        }
    }
}

/// @brief a helper function that reads the test datasets
/// @param ptr the pointer pointing to the char array
/// @param fname the name of the file
/// @param refLen the expected length of the sequence
void readFile(char* ptr, const char* fname, int& refLen)
{
    // https://www.ibm.com/docs/en/i/7.3?topic=functions-fgetc-read-character#fget
    FILE *stream;
    stream = fopen(fname,"r");
    if (!stream)
        printf("Could not open data file for reading, please select another length...\n");
    int i, ch;

    for (i = 0; (i < MAX_SEQ_LEN && ((ch = fgetc(stream)) != EOF) && (ch != '\n')); i++)
      *(ptr+i) = ch;

    refLen = i;
    *(ptr+i) = '\0';


    if (fclose(stream))
        perror("fclose error");
}


int main(int argc, char *argv[])
{

    if (argc != 3) return EXIT_FAILURE;

    char* ref = new char[MAX_SEQ_LEN];
    char* query = new char[MAX_SEQ_LEN];
    GpuTimer mainTimer, kernelTimer;
    
    int refLen, queryLen;

    // read file
    std::string refFileName, queryFileName;
    int lenSelect = std::stoi(argv[1]);
    std::ostringstream oss;
    oss << "../data/ref" << lenSelect << ".txt";
    refFileName = oss.str();
    oss.str("");
    oss << "../data/query" << lenSelect << ".txt";
    queryFileName = oss.str();

    readFile(ref, refFileName.c_str(), refLen);
    readFile(query, queryFileName.c_str(), queryLen);

    printf("*** Info ***\n");
    printf("lenSelect: %d\n", lenSelect);
    printf("ref sequence: %s\n", ref);
    printf("qry sequence: %s\n", query);
    printf("*** Info End ***\n");
    
    int* VP = new int[MAX_SEQ_LEN * MAX_BLOCK_SEQ_LEN];
    memset(VP, -1, MAX_SEQ_LEN * MAX_BLOCK_SEQ_LEN);
    // assume refLen == queryLen
    int numBlocks = 0;
    if ((refLen % MAX_BLOCK_SEQ_LEN) == 0){
        numBlocks = refLen / MAX_BLOCK_SEQ_LEN;
    } else {
        numBlocks = (refLen / MAX_BLOCK_SEQ_LEN) + 1;
    }

    mainTimer.Start();

    char* d_ref, * d_query;
    int* d_VP;
    cudaMalloc(&d_ref, sizeof(char) * refLen);
    cudaMalloc(&d_query, sizeof(char) * refLen);
    cudaMalloc(&d_VP, sizeof(int) * refLen * refLen);

    cudaMemcpy(d_ref, ref, sizeof(char) * refLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, sizeof(char) * refLen, cudaMemcpyHostToDevice);

    kernelTimer.Start();

    for (std::size_t itr = 0; itr < 1; itr++)
    {
        if (std::stoi(argv[2]) == 0)
            viterbi<<<1, 1, sizeof(float) * MAX_BLOCK_SEQ_LEN * MAX_BLOCK_SEQ_LEN * 3>>> (d_ref, d_query, refLen, d_VP);
        else if (std::stoi(argv[2]) == 1)
            parallel<<<numBlocks, MAX_BLOCK_SEQ_LEN, sizeof(float) * MAX_BLOCK_SEQ_LEN * MAX_BLOCK_SEQ_LEN * 3>>> (d_ref, d_query, refLen, queryLen, d_VP);
        // else
        //     parallel4<<<1, 4*refLen, sizeof(float) * refLen * refLen * 3>>> (refLen, d_VP);
    }
    
    kernelTimer.Stop();
    
    // copy the data from the device memory back to the host
    cudaMemcpy(ref, d_ref, sizeof(char) * refLen, cudaMemcpyDeviceToHost);
    cudaMemcpy(query, d_query, sizeof(char) * refLen, cudaMemcpyDeviceToHost);
    cudaMemcpy(VP, d_VP, sizeof(float) * refLen * refLen, cudaMemcpyDeviceToHost);

    // free cuda memory
    cudaFree(d_ref);
    cudaFree(d_query);

    mainTimer.Stop();

    // print the VP grid
    #if DEBUG
        printf("*** VP Host ***\n");
        for (int i = 0; i < refLen; i++)
        {
            for (int j = 0; j < refLen; j++)
            {
                printf("%d  ", *(VP+i*refLen+j));
            }
            printf("\n");
        }
    #endif

    printf("*** Performance Result ***\n");
    printf("The kernel ran in: %.4f msecs.\n", kernelTimer.Elapsed());
    printf("The code code ran in: %.4f msecs. (end-to-end)\n", mainTimer.Elapsed());

    printf("*** Performance End ***\n");

    // translate traceback
    unsigned int numM = 0, numJ = 0, numI = 0;
    std::deque<char> d;

    if (refLen > 64 || queryLen > 64)
    {
        for(int k = 0; k < numBlocks; k++)
        {
            int i = refLen-1 - k*MAX_BLOCK_SEQ_LEN, j = MAX_BLOCK_SEQ_LEN-1;
            while (i > refLen - (k+1)*MAX_BLOCK_SEQ_LEN || j > 0)
            {
                if (*(VP+i*MAX_BLOCK_SEQ_LEN+j) == 0)
                {
                    d.emplace_front('M');
                    ++numM;
                    --i, --j;
                }
                else if (*(VP+i*MAX_BLOCK_SEQ_LEN+j) == 1)
                {
                    d.emplace_front('J');
                    ++numJ;
                    --i;
                }
                else
                {
                    d.emplace_front('I');
                    ++numI;
                    --j;
                }
            }
            d.emplace_front('M');
            ++numM;
        }
    }
    else
    {
        int i = refLen-1, j = queryLen-1;
        while (i != 0 || j != 0)
        {
            if (*(VP+i*refLen+j) == 0)
            {
                d.emplace_front('M');
                ++numM;
                --i, --j;
            }
            else if (*(VP+i*refLen+j) == 1)
            {
                d.emplace_front('J');
                ++numJ;
                --i;
            }
            else
            {
                d.emplace_front('I');
                ++numI;
                --j;
            }
        }
        d.emplace_front('M');
        ++numM;
    }

    
    #if DEBUG
        auto itr_align = d.cbegin();
        std::cout << "*** Alignment Result ***\n";
        while (itr_align != d.cend())
        {
            std::cout << *itr_align << " ";
            ++itr_align;
        }
        std::cout << "\n*** Alignment End ***\n";
    #endif

    std::cout << "*** Identities ***\n";
    float identities = (float) 100*numM / queryLen;
    printf("%d/%d (%.2f%%)", numM, refLen, identities);
    std::cout << "\n*** Identities End ***\n";


    std::cout << "*** Alignment Result ***\n";
    auto itr = d.cbegin();
    int itrRef = 0;
    while (itr != d.cend() || itrRef < refLen)
    {
        if (itr != d.cend())
        {
            if (*itr == 'M' || *itr == 'J')
            {
                std::cout << *(ref+itrRef);
                ++itrRef;
            }
            else
            {
                std::cout << '-';
            }
            ++itr;
        }

        else
        {
            std::cout << *(ref+itrRef);
            ++itrRef;
        }
    }
    std::cout << "\n";
    itr = d.cbegin();
    int itrQuery = 0;
    while (itr != d.cend() || itrQuery < queryLen)
    {
        if (itr != d.cend())
        {
            if (*itr == 'M' || *itr == 'I')
            {
                std::cout << *(query+itrQuery);
                ++itrQuery;
            }
            else
            {
                std::cout << '-';
            }
            ++itr;
        }
        else
        {
            std::cout << *(query+itrQuery);
            ++itrQuery;
        }
    }
    std::cout << "\n*** Alignment End ***\n";
    
    // free memory
    delete ref;
    delete query;
    delete VP;
    
    return 0;
}
