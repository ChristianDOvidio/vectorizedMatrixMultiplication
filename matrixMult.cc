#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <immintrin.h>

//initialize passed block with random floats (-500 to 500)
void initialize(float *inputBlock, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        inputBlock[i] = -500. + (float)std::rand() / (float)(RAND_MAX / 1000.);
    }
}

bool checkMult(float *vectorInSeq, float *vectorInVec, int m)
{
    float vecOverSeq = 0.0;
    float seqOverVec = 0.0;
    for (int i = 0; i < m; i++)
    {
        vecOverSeq = vectorInVec[i] / vectorInSeq[i];
        seqOverVec = vectorInSeq[i] / vectorInVec[i];
        if (vecOverSeq > 1.5 || vecOverSeq < -1.5 || seqOverVec > 1.5 || seqOverVec < -1.5)
        {
            std::cout << "Sequential: " << vectorInSeq[i] << " Vectorized: " << vectorInVec[i] << std::endl;
            return false; //because of rounding of float, sums should just be very close to each other
            //only returns false if values are more than 1.5% different than each other
        }
    }
    return true;
}

void multSeq(float *vectorIn, float *matrix, int m, int n, float *vectorOut)
{
    float sum = 0.0;
    for (int i = 0; i < m; i++)
    { //number of rows
        sum = 0.0;
        for (int j = 0; j < n; j++)
        { //number of columns
            sum += matrix[j + (i * n)] * vectorIn[j];
        }
        vectorOut[i] = sum;
        //std::cout << "Seq sum is " << sum << std::endl;
    }
}

void matvec_simd(float *vectorIn, float *matrix, int m, int n, float *vectorOut)
{
    // create variables in AVX registers
    __m256 matrixReg;
    __m256 vectorReg;
    __m256 sumReg;
    // temporary registers to sum elements of register
    __m256 temp1;
    __m256 temp2;
    __m128 temp3;
    __m128 temp4;
    // output of vectorized code to add to sequential code
    float tempSum = 0.0;
    float sum = 0.0;
    float *ptr;
    // how to iterate over columns and rows
    int j = 0;
    int i = 0;
    int offset = 0;

    //number of vectorized operations
    int numIter = n / 8;
    //number of sequential operations
    int extraIter = n % 8;

    for (i = 0; i < m; i++)
    { //number of rows
        sum = 0.0;
        sumReg = _mm256_setzero_ps();
        for (j = 0, offset = 0; j < numIter; j++, offset += 8)
        {
            // load 256/32=8 values
            //std::cout << "Loading 8 matrix values starting from " << (0 + offset + (i * n)) << std::endl;
            if (extraIter == 0)
                matrixReg = _mm256_load_ps(matrix + offset + (i * n));
            else
                matrixReg = _mm256_loadu_ps(matrix + offset + (i * n));

            //std::cout << "Loading 8 vector values starting from " << (0 + offset) << std::endl;
            if (extraIter == 0)
                vectorReg = _mm256_load_ps(vectorIn + offset);
            else
                vectorReg = _mm256_loadu_ps(vectorIn + offset);

            //multiply these values
            matrixReg = _mm256_mul_ps(matrixReg, vectorReg);
            //increment sumReg with these new values
            sumReg = _mm256_add_ps(sumReg, matrixReg);
        }
        if (numIter > 0)
        { //only execute if you performed vectorized code
            //https://stackoverflow.com/a/18616679/4852830
            temp1 = _mm256_hadd_ps(sumReg, sumReg);
            temp2 = _mm256_hadd_ps(temp1, temp1);
            temp3 = _mm256_extractf128_ps(temp2, 1);
            temp4 = _mm_add_ss(_mm256_castps256_ps128(temp2), temp3);
            tempSum = _mm_cvtss_f32(temp4);
            //std::cout << "Vectorized sum is " << tempSum << std::endl;
            /*//understandable but slower
            float *ptr = (float *)&sumReg;
            tempSum = 0.0;
            for (int z = 0; z < 8; z++)
            {
                tempSum += ptr[z];
            }*/
        }
        if (extraIter > 0)
        { //only execute if there are leftover items to perform on
            for (j = offset; j < n; j++)
            { //resume where vectorized code left off
                //std::cout << "Loading matrix value from " << j + (i * n) << std::endl;
                //std::cout << "Loading vector value from " << j << std::endl;
                sum += matrix[j + (i * n)] * vectorIn[j];
            }
        }
        vectorOut[i] = sum + tempSum;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
        return 0;

    // define array size
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    //m-row*n-column matrix multiplied by n-row*1-column vector
    //results in a m-row*1-column vector
    if (n < 1 || m < 1)
        return 0;
    if (n * m > 100000000)
        return 0;

    //initialize rand
    std::srand(std::time(nullptr));

    // allocate aligned memory blocks
    float *vectorIn = (float *)_mm_malloc(n * sizeof(float), 32);
    float *vectorOutSeq = (float *)_mm_malloc(m * sizeof(float), 32);
    float *vectorOut = (float *)_mm_malloc(m * sizeof(float), 32);
    float *matrix = (float *)_mm_malloc(m * n * sizeof(float), 32);

    //initialize the input blocks
    initialize(vectorIn, n, 1);
    initialize(matrix, m, n);

    // find max sequentially
    auto start = std::chrono::high_resolution_clock::now();
    multSeq(vectorIn, matrix, m, n, vectorOutSeq);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "sequential: " << diff.count() << " sec\t"
              << "\t" << n << std::endl;

    // find max vectorially
    start = std::chrono::high_resolution_clock::now();
    matvec_simd(vectorIn, matrix, m, n, vectorOut);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "vectorized: " << diff.count() << " sec\t"
              << "\t" << n << std::endl;

    //check if results are the same
    if (!checkMult(vectorOutSeq, vectorOut, m))
    {
        std::cout << "Vector Incorrect !";
    }

    return 1;
}
