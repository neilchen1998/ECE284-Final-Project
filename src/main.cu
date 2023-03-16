#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <deque>

#include "timer.h"
#include "utils.h"

#define MAX_SEQ_LEN 64

#define DELTA 0.3
#define EPSILON 0.8

#define MATCH 0.9
#define MISMATCH 0.05
#define GAP 0.3

#define DEBUG 0

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

__global__ void parallel(char* ref, char* query, int refLen, int queryLen, int* VP)
{
    extern __shared__ float array[];
    float* VM = (float*)array;
    float* VI = (float*)&VM[refLen * queryLen];
    float* VJ = (float*)&VI[refLen * queryLen];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float currM = 0.0;
    float currI = 0.0;
    float currJ = 0.0;

    // initialize matrices
    for (std::size_t i = threadIdx.x; i < refLen * refLen; i += blockDim.x)
    {
        *(VM+i) = 0.0001;
        *(VI+i) = 0.0001;
        *(VJ+i) = 0.0001;
        if (idx == 0)   *VM = 1.0;
    }

    __syncthreads();

    // upper triangle
    for (std::size_t x = 0; x < refLen; x++)
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
                a = (1-2*DELTA) * *(VM+refLen*(i-1)+(j-1));
                b = (1-EPSILON) * *(VI+refLen*(i-1)+(j-1));
                c = (1-EPSILON) * *(VJ+refLen*(i-1)+(j-1));

                if (*(ref+i) == *(query+j))
                    currM  = MATCH * MAX(MAX(a, b), c);
                else
                    currM = MISMATCH * MAX(MAX(a, b), c);

                *(VM+refLen*i+j) = currM;
            }

            // VI
            if (j > 0)
            {
                //a = DELTA * *(VM+refLen*(i-1)+j);
                //b = EPSILON * *(VI+refLen*(i-1)+j);
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
                *(VP+refLen*i+j) = 0;
            else if (I > J)
                *(VP+refLen*i+j) = 2;
            else
                *(VP+refLen*i+j) = 1;
        }

        __syncthreads();
    }

    // lower triangle
    for (std::size_t x = refLen; x < 2*refLen-1; x++)
    {
        if ( idx <= ((2*refLen -2) - x))
        {
            // index is (x-N+1+y, N-1-y)
            std::size_t y = idx;
            int i = x - refLen + 1 + y, j = refLen - 1 - y;

            float a, b, c;

            // VM
            if (i > 0 && j > 0)
            {
                a = (1-2*DELTA) * *(VM+refLen*(i-1)+(j-1));
                b = (1-EPSILON) * *(VI+refLen*(i-1)+(j-1));
                c = (1-EPSILON) * *(VJ+refLen*(i-1)+(j-1));

                if (*(ref+i) == *(query+j))
                    currM  = MATCH * MAX(MAX(a, b), c);
                else
                    currM = MISMATCH * MAX(MAX(a, b), c);

                *(VM+refLen*i+j) = currM;
            }

            // VI
            if (j > 0)
            {
                a = DELTA * *(VM+refLen*i+(j-1));
                b = EPSILON * *(VJ+refLen*i+(j-1));
                currI = GAP * MAX(a, b);
                *(VI+refLen*i+j) = currI;
            }

            // VD
            if (i > 0)
            {
                a = DELTA * *(VM+refLen*(i-1)+j);
                b = EPSILON * *(VI+refLen*(i-1)+j);
                currJ = GAP * MAX(a, b);
                *(VJ+refLen*i+j) = currJ;

            }

            // VP
            float M = currM, I = currI, J = currJ;
            if (M >= I && M >= J)
                *(VP+refLen*i+j) = 0;
            else if (I > J)
                *(VP+refLen*i+j) = 2;
            else
                *(VP+refLen*i+j) = 1;
        }

        __syncthreads();
    }

    __syncthreads();


    #if DEBUG
        if (idx == 0)
        {
            printf("*** VM Parallel ***\n");
            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    printf("%.6f  ", *(VM+i*refLen+j));
                }
                printf("\n");
            }
            printf("*** End ***\n");

            printf("*** VJ Parallel ***\n");
            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    printf("%.6f  ", *(VJ+i*refLen+j));
                }
                printf("\n");
            }
            printf("*** End ***\n");

            printf("*** VI Parallel ***\n");
            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    printf("%.6f  ", *(VI+i*refLen+j));
                }
                printf("\n");
            }
            printf("*** End ***\n");
        }
    #endif

}

__global__ void viterbi(char* ref, char* query, int refLen, int* VP)
{
    extern __shared__ float array[];
    float* VM = (float*)array;
    float* VI = (float*)&VM[refLen * refLen];
    float* VJ = (float*)&VI[refLen * refLen];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize matrices
    for (std::size_t i = threadIdx.x; i < refLen * refLen; i += blockDim.x)
    {
        *(VM+i) = 0.0001;
        *(VI+i) = 0.0001;
        *(VJ+i) = 0.0001;
        if (idx == 0)   *VM = 1.0;
    }

    __syncthreads();

    if (idx == 0)
    {
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

                    if (*(ref+i) == *(query+j))
                        *(VM+refLen*i+j)  = MATCH * MAX(MAX(a, b), c);
                    else
                        *(VM+refLen*i+j) = MISMATCH * MAX(MAX(a, b), c);

                    }

                // VI
                if (i > 0)
                {
                    a = DELTA * *(VM+refLen*(i-1)+j);
                    b = EPSILON * *(VI+refLen*(i-1)+j);

                    *(VI+refLen*i+j) = MISMATCH * MAX(a, b);
                }

                // VD
                if (j > 0)
                {
                    a = DELTA * *(VM+refLen*i+(j-1));
                    b = EPSILON * *(VJ+refLen*i+(j-1));
                    *(VJ+refLen*i+j) = MISMATCH * MAX(a, b);
                }

                // VP
                float M = *(VM+refLen*i+j), I = *(VI+refLen*i+j), J = *(VJ+refLen*i+j);
                if (M >= I && M >= J)
                    *(VP+refLen*i+j) = 0;
                else if (I > J)
                    *(VP+refLen*i+j) = 1;
                else
                    *(VP+refLen*i+j) = 2;
            }
        }
    }

    __syncthreads();

    #if DEBUG
        if (idx == 0)
        {
            printf("*** VM Parallel ***\n");
            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    printf("%.6f  ", *(VM+i*refLen+j));
                }
                printf("\n");
            }
            printf("*** End ***\n");

            printf("*** VJ Parallel ***\n");
            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    printf("%.6f  ", *(VJ+i*refLen+j));
                }
                printf("\n");
            }
            printf("*** End ***\n");

            printf("*** VI Parallel ***\n");
            for (int i = 0; i < refLen; i++)
            {
                for (int j = 0; j < refLen; j++)
                {
                    printf("%.6f  ", *(VI+i*refLen+j));
                }
                printf("\n");
            }
            printf("*** End ***\n");
        }
    #endif

    __syncthreads();
}

void readFile(char* ptr, const char* fname, int& refLen)
{
    // https://www.ibm.com/docs/en/i/7.3?topic=functions-fgetc-read-character#fget
    FILE *stream;
    stream = fopen(fname,"r");
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
    std::string refFileName("../data/ref8.txt"), queryFileName("../data/query8.txt");

    int lenSelect = std::stoi(argv[1]);
    if (lenSelect != 8)
    {
        std::ostringstream oss;
        oss << "../data/" << "ref" << lenSelect << ".txt";
        refFileName = oss.str();
        oss.str("");
        oss << "../data/" << "query" << lenSelect << ".txt";
        queryFileName = oss.str();
    }

    readFile(ref, refFileName.c_str(), refLen);
    readFile(query, queryFileName.c_str(), queryLen);

    printf("*** Info ***\n");
    printf("lenSelect: %d\n", lenSelect);
    printf("ref sequence: %s\n", ref);
    printf("qry sequence: %s\n", query);
    printf("*** Info End ***\n");
    
    int* VP = new int[MAX_SEQ_LEN * MAX_SEQ_LEN];
    memset(VP, -1, MAX_SEQ_LEN * MAX_SEQ_LEN);

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
            viterbi<<<1, 1, sizeof(float) * refLen * refLen * 3>>> (d_ref, d_query, refLen, d_VP);
        else if (std::stoi(argv[2]) == 1)
            parallel<<<1, refLen, sizeof(float) * refLen * queryLen * 3>>> (d_ref, d_query, refLen, queryLen, d_VP);
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
    int i = refLen-1, j = queryLen-1;
    unsigned int numM = 0, numJ = 0, numI = 0;
    std::deque<char> d;
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
    // auto itr = d.cbegin();
    // std::cout << "*** Alignment Result ***\n";
    // while (itr != d.cend())
    // {
    //     std::cout << *itr << " ";
    //     ++itr;
    // }
    // std::cout << "\n*** Alignment End ***\n";

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
    
    delete ref;
    delete query;
    delete VP;
    
    return 0;
}

