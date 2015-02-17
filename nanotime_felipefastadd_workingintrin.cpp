/*  nanotime_felipefastadd_workingintrin.cpp
    Written 17 February 2015 by Daniel Richman. 
    Adds floats speedily (though loses precision)
    to test C++ performance vs Julia for Felipe's
    assignment for 18.335. 
    
    We use AVX intrinsics to perform 8 parallel adds. 
*/

#include <iostream>
#include <random>
#include <chrono>
#include <stdlib.h> // for posix_memalign
#include <immintrin.h>

using namespace std;
using namespace std::chrono;

std::default_random_engine generator;
std::uniform_real_distribution<float> distribution(0.0,1.0);

const long QTY = 1000000000;
float * makeNumbers() {
    srand(time(NULL));
    
    float *n;
    posix_memalign((void **) &n, 16, sizeof(float) * QTY); // allocates our memory aligned to 16-bit for _mm_store_ps speedup
    for (long j = 0; j < QTY; j++) {
        n[j] = distribution(generator);
    }
    return n;
}

int main() {

    float *nums = makeNumbers();
    float total = 0.;

    __m256 partial_sums;
    
    // make sure partial_sums starts at 0
    partial_sums = _mm256_xor_ps(partial_sums, partial_sums);
        
    // perform adds
    auto start = std::chrono::steady_clock::now();
    
    for (long j = 0; j < QTY; j+= 8) {
        partial_sums = _mm256_add_ps(partial_sums, * (__m256 *) (nums + j));
    }
    
    auto end = std::chrono::steady_clock::now();
    
    // extract 8 final floats and print values
    float answers[8]; // __attribute__((__aligned__(16))); // alignment unnecessary
    
    _mm256_store_ps(answers, partial_sums);
    
    float final_sum = 0;
    
    for (int j = 0; j < 8; j++) {
        final_sum += answers[j];
        cout << j << ": " << answers[j] << endl;
    }
    
    cout << "SUM: " << final_sum << endl;
    
    auto elapsed = end - start;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cout << "time taken (sec): " << (double) nanoseconds * 1e-9 << endl;
    
    
    // Now we compute the correct answer using double-precision. 
    // On my machine it's about 5.0 * 10^8, which makes sense, since we have
    // 10^9 random numbers with a uniform distribution centered about 0.5. 
    // The error from 32-bit fp is substantial if we add stupidly. 
    double real_ans = 0;
    
    for (long j = 0; j < QTY; j++)
        real_ans += nums[j];
    
    cout << real_ans << endl;
}
