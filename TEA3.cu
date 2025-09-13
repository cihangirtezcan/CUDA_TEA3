#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include <inttypes.h>
#include <string.h>

#define BLOCKS				1024
#define THREADS				256  // Cannot be less than 256 because at the beginning of each kernel the first 256 threads in a block copies 256 S-box values from global memory to the shared memory
#define BLOCKSLOG				10
#define THREADSLOG				8
int choice = 0; int trials = 0;
const uint16_t g_awTea3LutA[8] = { 0x92A7, 0xA761, 0x974C, 0x6B8C, 0x29CE, 0x176C, 0x39D4, 0x7463 };
const uint16_t g_awTea3LutB[8] = { 0x9D58, 0xA46D, 0x176C, 0x79C4, 0xC62B, 0xB2C9, 0x4D93, 0x2E93 };
const uint8_t g_abTea3Sbox[256] = {
    0x7D, 0xBF, 0x7B, 0x92, 0xAE, 0x7C, 0xF2, 0x10, 0x5A, 0x0F, 0x61, 0x7A, 0x98, 0x76, 0x07, 0x64,
    0xEE, 0x89, 0xF7, 0xBA, 0xC2, 0x02, 0x0D, 0xE8, 0x56, 0x2E, 0xCA, 0x58, 0xC0, 0xFA, 0x2A, 0x01,
    0x57, 0x6E, 0x3F, 0x4B, 0x9C, 0xDA, 0xA6, 0x5B, 0x41, 0x26, 0x50, 0x24, 0x3E, 0xF8, 0x0A, 0x86,
    0xB6, 0x5C, 0x34, 0xE9, 0x06, 0x88, 0x1F, 0x39, 0x33, 0xDF, 0xD9, 0x78, 0xD8, 0xA8, 0x51, 0xB2,
    0x09, 0xCD, 0xA1, 0xDD, 0x8E, 0x62, 0x69, 0x4D, 0x23, 0x2B, 0xA9, 0xE1, 0x53, 0x94, 0x90, 0x1E,
    0xB4, 0x3B, 0xF9, 0x4E, 0x36, 0xFE, 0xB5, 0xD1, 0xA2, 0x8D, 0x66, 0xCE, 0xB7, 0xC4, 0x60, 0xED,
    0x96, 0x4F, 0x31, 0x79, 0x35, 0xEB, 0x8F, 0xBB, 0x54, 0x14, 0xCB, 0xDE, 0x6B, 0x2D, 0x19, 0x82,
    0x80, 0xAC, 0x17, 0x05, 0xFF, 0xA4, 0xCF, 0xC6, 0x6F, 0x65, 0xE6, 0x74, 0xC8, 0x93, 0xF4, 0x7E,
    0xF3, 0x43, 0x9F, 0x71, 0xAB, 0x9A, 0x0B, 0x87, 0x55, 0x70, 0x0C, 0xAD, 0xCC, 0xA5, 0x44, 0xE7,
    0x46, 0x45, 0x03, 0x30, 0x1A, 0xEA, 0x67, 0x99, 0xDB, 0x4A, 0x42, 0xD7, 0xAA, 0xE4, 0xC2, 0xD5,
    0xF0, 0x77, 0x20, 0xC3, 0x3C, 0x16, 0xB9, 0xE2, 0xEF, 0x6C, 0x3D, 0x1B, 0x22, 0x84, 0x2F, 0x81,
    0x1D, 0xB1, 0x3A, 0xE5, 0x73, 0x40, 0xD0, 0x18, 0xC7, 0x6A, 0x9E, 0x91, 0x48, 0x27, 0x95, 0x72,
    0x68, 0x0E, 0x00, 0xFC, 0xC5, 0x5F, 0xF1, 0xF5, 0x38, 0x11, 0x7F, 0xE3, 0x5E, 0x13, 0xAF, 0x37,
    0xE0, 0x8A, 0x49, 0x1C, 0x21, 0x47, 0xD4, 0xDC, 0xB0, 0xEC, 0x83, 0x28, 0xB8, 0xF6, 0xA7, 0xC9,
    0x63, 0x59, 0xBD, 0x32, 0x85, 0x08, 0xBE, 0xD3, 0xFD, 0x4C, 0x2C, 0xFB, 0xA0, 0xC1, 0x9D, 0xB3,
    0x52, 0x8C, 0x5D, 0x29, 0x6D, 0x04, 0xBC, 0x25, 0x15, 0x8B, 0x12, 0x9B, 0xD6, 0x75, 0xA3, 0x97
};
static uint64_t tea3_compute_iv(uint32_t dwFrameNumbers) {
    uint32_t dwXorred = dwFrameNumbers ^ 0xC43A7D51;
    dwXorred = (dwXorred << 8) | (dwXorred >> 24); // rotate left -> translated to single rol instruction
    uint64_t qwIv = ((uint64_t)dwFrameNumbers << 32) | dwXorred;
    return (qwIv >> 8) | (qwIv << 56); // rotate right
}
__device__ static uint64_t gpu_tea3_compute_iv(uint32_t dwFrameNumbers) {
    uint32_t dwXorred = dwFrameNumbers ^ 0xC43A7D51;
    dwXorred = (dwXorred << 8) | (dwXorred >> 24); // rotate left -> translated to single rol instruction
    uint64_t qwIv = ((uint64_t)dwFrameNumbers << 32) | dwXorred;
    return (qwIv >> 8) | (qwIv << 56); // rotate right
}
static uint8_t tea3_state_word_to_newbyte(uint16_t wSt, const uint16_t* awLut) {
    uint8_t bSt0 = wSt;
    uint8_t bSt1 = wSt >> 8;

    uint8_t bDist;
    uint8_t bOut = 0;

    for (int i = 0; i < 8; i++) {
        // taps on bit 5,6 for bSt0 and bit 5,6 for bSt1
        bDist = ((bSt0 >> 5) & 3) | ((bSt1 >> 3) & 12);
        if (awLut[i] & (1 << bDist)) {
            bOut |= 1 << i;
        }

        // rotate one position
        bSt0 = ((bSt0 >> 1) | (bSt0 << 7));
        bSt1 = ((bSt1 >> 1) | (bSt1 << 7));
    }

    return bOut;
}
__device__ static uint8_t gpu_tea3_state_word_to_newbyte2(uint16_t wSt, const uint32_t* awLut, int warpThreadIndex) {
    uint8_t bSt0 = wSt;
    uint8_t bSt1 = wSt >> 8;
    uint8_t bDist;
    uint8_t bOut = 0;

    for (int i = 0; i < 7; i++) {
        // taps on bit 5,6 for bSt0 and bit 5,6 for bSt1
        bDist = ((bSt0 >> 5) & 3) | ((bSt1 >> 3) & 12);
        if (awLut[32 * i + warpThreadIndex] & (1 << bDist)) {            bOut |= 1 << i;        }
        // rotate one position
        bSt0 = ((bSt0 >> 1) | (bSt0 << 7));
        bSt1 = ((bSt1 >> 1) | (bSt1 << 7));
    }
    // taps on bit 5,6 for bSt0 and bit 5,6 for bSt1
    bDist = ((bSt0 >> 5) & 3) | ((bSt1 >> 3) & 12);
    if (awLut[7 * 32 + warpThreadIndex] & (1 << bDist)) {        bOut |= 1 << 7;    }

    return bOut;
}
__device__ static uint8_t gpu_tea3_state_word_to_newbyte(uint16_t wSt, const uint16_t* awLut) {
    uint8_t bSt0 = wSt;
    uint8_t bSt1 = wSt >> 8;

    uint8_t bDist;
    uint8_t bOut = 0;

    /*   for (int i = 0; i < 8; i++) {
           // taps on bit 5,6 for bSt0 and bit 5,6 for bSt1
           bDist = ((bSt0 >> 5) & 3) | ((bSt1 >> 3) & 12);
           if (awLut[i] & (1 << bDist)) {
               bOut |= 1 << i;
           }
           // rotate one position
           bSt0 = ((bSt0 >> 1) | (bSt0 << 7));
           bSt1 = ((bSt1 >> 1) | (bSt1 << 7));
       }*/
    for (int i = 0; i < 7; i++) {
        // taps on bit 5,6 for bSt0 and bit 5,6 for bSt1
        bDist = ((bSt0 >> 5) & 3) | ((bSt1 >> 3) & 12);
        if (awLut[i] & (1 << bDist)) { bOut |= 1 << i; }
        // rotate one position
        bSt0 = ((bSt0 >> 1) | (bSt0 << 7));
        bSt1 = ((bSt1 >> 1) | (bSt1 << 7));
    }
    // taps on bit 5,6 for bSt0 and bit 5,6 for bSt1
    bDist = ((bSt0 >> 5) & 3) | ((bSt1 >> 3) & 12);
    if (awLut[7] & (1 << bDist)) { bOut |= 1 << 7; }

    return bOut;
}
static uint8_t tea3_reorder_state_byte(uint8_t bStByte) {
    // simple re-ordering of bits
    uint8_t bOut = 0;
    bOut |= ((bStByte << 6) & 0x40);
    bOut |= ((bStByte << 1) & 0x20);
    bOut |= ((bStByte << 2) & 0x98);
    bOut |= ((bStByte >> 4) & 0x04);
    bOut |= ((bStByte >> 3) & 0x01);
    bOut |= ((bStByte >> 6) & 0x02);
    return bOut;
}
__device__ static uint8_t gpu_tea3_reorder_state_byte(uint8_t bStByte) {
    // simple re-ordering of bits
    uint8_t bOut = 0;
    bOut |= ((bStByte << 6) & 0x40);
    bOut |= ((bStByte << 1) & 0x20);
    bOut |= ((bStByte << 2) & 0x98);
    bOut |= ((bStByte >> 4) & 0x04);
    bOut |= ((bStByte >> 3) & 0x01);
    bOut |= ((bStByte >> 6) & 0x02);
    return bOut;
}
void tea3(uint32_t dwFrameNumbers, uint8_t* lpKey, uint32_t dwNumKsBytes, uint8_t* lpKsOut) {
    uint8_t abKeyReg[10];
    uint32_t dwNumSkipRounds = 51;

    // init registers
    uint64_t qwIvReg = tea3_compute_iv(dwFrameNumbers);
    memcpy(abKeyReg, lpKey, 10);

    for (int i = 0; i < dwNumKsBytes; i++) {
        for (int j = 0; j < dwNumSkipRounds; j++) {
            // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
            uint8_t bSboxOut = g_abTea3Sbox[abKeyReg[7] ^ abKeyReg[2]] ^ abKeyReg[0];
            memmove(abKeyReg, abKeyReg + 1, 9);
            abKeyReg[9] = bSboxOut;

            // Step 2: Compute 3 bytes derived from current state
            uint8_t bDerivByte12 = tea3_state_word_to_newbyte((qwIvReg >> 8) & 0xffff, g_awTea3LutA);
            uint8_t bDerivByte56 = tea3_state_word_to_newbyte((qwIvReg >> 40) & 0xffff, g_awTea3LutB);
            uint8_t bReordByte4 = tea3_reorder_state_byte((qwIvReg >> 32) & 0xff);

            // Step 3: Combine current state with state derived values, and xor in key derived sbox output
            uint8_t bNewByte = ((qwIvReg >> 56) ^ bReordByte4 ^ bDerivByte12 ^ bSboxOut) & 0xff;
            uint8_t bMixByte = bDerivByte56;

            // Step 4: Update lfsr: leftshift 8, feed/mix in previously generated bytes
            qwIvReg = ((qwIvReg << 8) ^ ((uint64_t)bMixByte << 40)) | bNewByte;
        }
        lpKsOut[i] = (qwIvReg >> 56);
        dwNumSkipRounds = 19;
    }
}
__global__ void tea3_exhaustive(uint32_t dwFrameNumbers, uint32_t dwNumKsBytes, uint8_t* lpKsOut_d, uint16_t *gpu_awTea3LutA_d, uint16_t* gpu_awTea3LutB_d, uint8_t *gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint16_t gpu_awTea3LutA[8];
    __shared__ uint16_t gpu_awTea3LutB[8];
    __shared__ uint8_t gpu_abTea3Sbox[256];
    __shared__ uint8_t lpKsOut[10];
    if (threadIdx.x < 256) {
        gpu_abTea3Sbox[threadIdx.x] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 10) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
        if (threadIdx.x < 8) {
            gpu_awTea3LutA[threadIdx.x] = gpu_awTea3LutA_d[threadIdx.x];
            gpu_awTea3LutB[threadIdx.x] = gpu_awTea3LutB_d[threadIdx.x];
        }
    }
    __syncthreads();
    uint16_t key_right; // rightmost 16 bits
    uint64_t key_left; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
//    threadIndex = threadIndex << (64 - BLOCKSLOG - THREADSLOG);
    uint64_t IV = gpu_tea3_compute_iv(dwFrameNumbers);
    for (uint16_t trial = 0; trial < trials; trial++) {
        uint32_t dwNumSkipRounds = 51;
        uint64_t qwIvReg = IV;
        key_right = trial; // rightmost 16 bits
        key_left = threadIndex; // leftmost 64 bits
        int flag = 1;

        for (int i = 0; (i < dwNumKsBytes) && flag==1; i++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint8_t bSboxOut = gpu_abTea3Sbox[(key_left & 0xff) ^ ((key_left >> 40) & 0xff)] ^ (key_left >> 56);
                key_left = (key_left << 8) | (key_right >> 8);
                key_right = (key_right << 8) | bSboxOut;
                // Step 2: Compute 3 bytes derived from current state
                uint8_t bDerivByte12 = gpu_tea3_state_word_to_newbyte((qwIvReg >> 8) & 0xffff, gpu_awTea3LutA);
                uint8_t bDerivByte56 = gpu_tea3_state_word_to_newbyte((qwIvReg >> 40) & 0xffff, gpu_awTea3LutB);
                uint8_t bReordByte4 = gpu_tea3_reorder_state_byte((qwIvReg >> 32) & 0xff);
                // Step 3: Combine current state with state derived values, and xor in key derived sbox output
                uint8_t bNewByte = ((qwIvReg >> 56) ^ bReordByte4 ^ bDerivByte12 ^ bSboxOut) & 0xff;
                uint8_t bMixByte = bDerivByte56;
                // Step 4: Update lfsr: leftshift 8, feed/mix in previously generated bytes
                qwIvReg = ((qwIvReg << 8) ^ ((uint64_t)bMixByte << 40)) | bNewByte;
            }
 //           lpKsOut[i] = (qwIvReg >> 56);
            if ((qwIvReg >> 56) != lpKsOut[i]) flag=0;
//            if (threadIndex == 0 && trial==0) printf("%llx %x\n", qwIvReg >> 56, lpKsOut[i]);
            dwNumSkipRounds = 19;
        }
        if (flag == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial;    }
//        if (flag == 1) printf("Hello world %llu\n",threadIndex);

    }
}
__global__ void tea3_exhaustive_0conflict(uint32_t dwFrameNumbers, uint32_t dwNumKsBytes, uint8_t* lpKsOut_d, uint16_t* gpu_awTea3LutA_d, uint16_t* gpu_awTea3LutB_d, uint8_t* gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint32_t gpu_awTea3LutA[256];
    __shared__ uint32_t gpu_awTea3LutB[256];
    __shared__ uint32_t gpu_abTea3Sbox[256][32];
    __shared__ uint8_t lpKsOut[10];
    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 256) {
        for (int i = 0; i < 32; i++) gpu_abTea3Sbox[threadIdx.x][i] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 10) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
        if (threadIdx.x < 8) {
            for (int i = 0; i < 32; i++) {
                gpu_awTea3LutA[32 * threadIdx.x + i] = gpu_awTea3LutA_d[threadIdx.x];
                gpu_awTea3LutB[32 * threadIdx.x + i] = gpu_awTea3LutB_d[threadIdx.x];
            }
        }
    }
    __syncthreads();  
    uint16_t key_right; // rightmost 16 bits
    uint64_t key_left; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    uint64_t IV = gpu_tea3_compute_iv(dwFrameNumbers);

    for (int trial = 0; trial < trials; trial++) {
        uint32_t dwNumSkipRounds = 51;
        uint64_t qwIvReg = IV;
        key_right = trial; // rightmost 16 bits
        key_left = threadIndex; // leftmost 64 bits
        int flag = 1;
        for (int i = 0; (i < dwNumKsBytes) && flag == 1; i++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint8_t bSboxOut = gpu_abTea3Sbox[(key_left & 0xff) ^ ((key_left >> 40) & 0xff)][warpThreadIndex] ^ (key_left >> 56);
                key_left = (key_left << 8) ^ (key_right >> 8);
                key_right = (key_right << 8) ^ bSboxOut;
                // Step 2: Compute 3 bytes derived from current state
                uint8_t bDerivByte12 = gpu_tea3_state_word_to_newbyte2((qwIvReg >> 8) & 0xffff, gpu_awTea3LutA,  warpThreadIndex);
                uint8_t bDerivByte56 = gpu_tea3_state_word_to_newbyte2((qwIvReg >> 40) & 0xffff, gpu_awTea3LutB, warpThreadIndex);
                uint8_t bReordByte4 = gpu_tea3_reorder_state_byte((qwIvReg >> 32) & 0xff);
                // Step 3: Combine current state with state derived values, and xor in key derived sbox output
                uint8_t bNewByte = ((qwIvReg >> 56) ^ bReordByte4 ^ bDerivByte12 ^ bSboxOut) & 0xff;
                uint8_t bMixByte = bDerivByte56;
                // Step 4: Update lfsr: leftshift 8, feed/mix in previously generated bytes
                qwIvReg = ((qwIvReg << 8) ^ ((uint64_t)bMixByte << 40)) | bNewByte;
            }
            if ((qwIvReg >> 56) != lpKsOut[i]) flag = 0;
            dwNumSkipRounds = 19;
        }
        if (flag == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial; }
    }
}
__global__ void tea3_exhaustive_1conflict(uint32_t dwFrameNumbers, uint32_t dwNumKsBytes, uint8_t* lpKsOut_d, uint16_t* gpu_awTea3LutA_d, uint16_t* gpu_awTea3LutB_d, uint8_t* gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint32_t gpu_awTea3LutA[256];
    __shared__ uint32_t gpu_awTea3LutB[256];
    __shared__ uint8_t gpu_abTea3Sbox[256];
    __shared__ uint8_t lpKsOut[10];
    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 256) {
        gpu_abTea3Sbox[threadIdx.x] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 10) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
        if (threadIdx.x < 8) {
            for (int i = 0; i < 32; i++) {
                gpu_awTea3LutA[32 * threadIdx.x + i] = gpu_awTea3LutA_d[threadIdx.x];
                gpu_awTea3LutB[32 * threadIdx.x + i] = gpu_awTea3LutB_d[threadIdx.x];
            }
        }
    }
    __syncthreads();
    uint16_t key_right; // rightmost 16 bits
    uint64_t key_left; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    uint64_t IV = gpu_tea3_compute_iv(dwFrameNumbers);
    for (int trial = 0; trial < trials; trial++) {
        uint64_t qwIvReg = IV;
        uint32_t dwNumSkipRounds = 51;
        key_right = trial; // rightmost 16 bits
        key_left = threadIndex; // leftmost 64 bits
        int flag = 1;
        for (int i = 0; (i < dwNumKsBytes) && flag == 1; i++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint8_t bSboxOut = gpu_abTea3Sbox[(key_left & 0xff) ^ ((key_left >> 40) & 0xff)] ^ (key_left >> 56);
                key_left = (key_left << 8) ^ (key_right >> 8);
                key_right = (key_right << 8) ^ bSboxOut;
                // Step 2: Compute 3 bytes derived from current state
                uint8_t bDerivByte12 = gpu_tea3_state_word_to_newbyte2((qwIvReg >> 8) & 0xffff, gpu_awTea3LutA, warpThreadIndex);
                uint8_t bDerivByte56 = gpu_tea3_state_word_to_newbyte2((qwIvReg >> 40) & 0xffff, gpu_awTea3LutB, warpThreadIndex);
                uint8_t bReordByte4 = gpu_tea3_reorder_state_byte((qwIvReg >> 32) & 0xff);
                // Step 3: Combine current state with state derived values, and xor in key derived sbox output
                uint8_t bNewByte = ((qwIvReg >> 56) ^ bReordByte4 ^ bDerivByte12 ^ bSboxOut) & 0xff;
                uint8_t bMixByte = bDerivByte56;
                // Step 4: Update lfsr: leftshift 8, feed/mix in previously generated bytes
                qwIvReg = ((qwIvReg << 8) ^ ((uint64_t)bMixByte << 40)) | bNewByte;
            }
            if ((qwIvReg >> 56) != lpKsOut[i]) flag = 0;
            dwNumSkipRounds = 19;
        }
        if (flag == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial; }
    }
}
// tea3_exhaustive_bitsliced kernel keeps each bit of the state in a byte (8-bit) variable
// These 8-bit variables keep 8 different states so one encryption performed by a thread actually means 8 encryptions in parallel
__global__ void tea3_exhaustive_bitsliced(uint8_t *reg, uint32_t dwNumKsBytes, uint8_t* lpKsOut_d, uint8_t* gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint8_t gpu_abTea3Sbox[256];
    __shared__ uint8_t lpKsOut[80];
    if (threadIdx.x < 256) {
        gpu_abTea3Sbox[threadIdx.x] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 80) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
    }
    __syncthreads();
    uint16_t key_right[8]; // rightmost 16 bits
    uint64_t key_left[8]; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    uint8_t r[64];
    uint8_t bDerivByte12[8], bDerivByte56[8], bReordByte4[8], bNewByte[8];
    int flag[8];
//    if (threadIndex == 0) for (int i = 0; i < 3; i++) printf("Reg %d: %x\n", i, reg[i]);
    //   threadIndex = threadIndex << (64 - BLOCKSLOG - THREADSLOG);    
    for (int trial = 0; trial < trials; trial++) {
        uint32_t dwNumSkipRounds = 51;
        for (int i = 0; i < 64; i++) r[i] = reg[i];
        for (uint64_t i = 0; i < 8; i++) {
            key_right[i] = trial; // rightmost 16 bits
            key_left[i] = (threadIndex << (64 - BLOCKSLOG - THREADSLOG)) ^ (i<<16); // leftmost 64 bits
            flag[i] = 1;
        }
        int flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7];
        for (int w = 0; (w < dwNumKsBytes) && flag_overall == 1; w++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint8_t bSboxOut[8];
                for (int i = 0; i < 8; i++) {
                    bSboxOut[i] = gpu_abTea3Sbox[(key_left[i] & 0xff) ^ ((key_left[i] >> 40) & 0xff)] ^ (key_left[i] >> 56);
                    key_left[i] = (key_left[i] << 8) ^ (key_right[i] >> 8);
                    key_right[i] = (key_right[i] << 8) ^ bSboxOut[i];
                }
 //               if (threadIndex == 0) for (int i = 0; i < 8; i++) printf("%llx %x\n",key_left[i], key_right[i]); // keys are generated correctly 
 //               if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 64; i++) printf("%02x\n", reg[i]); printf("\n"); }
                // Step 2: Compute 3 bytes derived from current state
/*                bDerivByte12[7] = r[45] & r[46] & r[53] ^ r[45] & r[46] ^ r[45] & r[53] & r[54] ^ r[45] & r[53] ^ r[45] & r[54] ^ r[46] & r[53] & r[54] ^ r[53] ^ r[54] ^ 0xff;
                bDerivByte12[6] = r[46] & r[47] & r[54] ^ r[46] & r[47] ^ r[46] & r[54] & r[55] ^ r[46] & r[55] ^ r[46] ^ r[47] & r[55] ^ r[47] ^ r[54] ^ 0xff;
                bDerivByte12[5] = r[40] & r[47] & r[48] ^ r[40] & r[47] & r[55] ^ r[40] & r[48] & r[55] ^ r[40] & r[48] ^ r[40] ^ r[47] & r[48] & r[55] ^ r[48];
                bDerivByte12[4] = r[40] & r[41] & r[48] ^ r[40] & r[41] & r[49] ^ r[40] & r[48] & r[49] ^ r[41] & r[48] & r[49] ^ r[41] & r[48] ^ r[41] ^ r[48] & r[49] ^ r[49];
                bDerivByte12[3] = r[41] & r[42] & r[49] ^ r[41] & r[42] & r[50] ^ r[41] & r[42] ^ r[41] & r[49] & r[50] ^ r[41] & r[49] ^ r[41] ^ r[42] & r[49] & r[50] ^ r[42] ^ r[49] & r[50] ^ r[50];
                bDerivByte12[2] = r[42] & r[43] & r[51] ^ r[42] & r[50] ^ r[43] & r[50] & r[51] ^ r[43] & r[51] ^ r[43] ^ r[51];
                bDerivByte12[1] = r[43] & r[44] & r[52] ^ r[43] & r[44] ^ r[43] & r[51] ^ r[43] & r[52] ^ r[44] & r[51] & r[52] ^ r[44] & r[51] ^ r[44] ^ r[51] & r[52] ^ r[51] ^ r[52];
                bDerivByte12[0] = r[44] & r[45] & r[53] ^ r[44] & r[52] & r[53] ^ r[44] & r[52] ^ r[45] & r[52] & r[53] ^ r[45] ^ r[52] ^ r[53] ^ 0xff;*/ //old

                bDerivByte12[7] = r[50] & r[49] & r[42] ^ r[50] & r[49] ^ r[50] & r[42] & r[41] ^ r[50] & r[42] ^ r[50] & r[41] ^ r[49] & r[42] & r[41] ^ r[42] ^ r[41] ^ 0xff;
                bDerivByte12[6] = r[49] & r[48] & r[41] ^ r[49] & r[48] ^ r[49] & r[41] & r[40] ^ r[49] & r[40] ^ r[49] ^ r[48] & r[40] ^ r[48] ^ r[41] ^ 0xff;
                bDerivByte12[5] = r[55] & r[48] & r[47] ^ r[55] & r[48] & r[40] ^ r[55] & r[47] & r[40] ^ r[55] & r[47] ^ r[55] ^ r[48] & r[47] & r[40] ^ r[47];
                bDerivByte12[4] = r[55] & r[54] & r[47] ^ r[55] & r[54] & r[46] ^ r[55] & r[47] & r[46] ^ r[54] & r[47] & r[46] ^ r[54] & r[47] ^ r[54] ^ r[47] & r[46] ^ r[46];
                bDerivByte12[3] = r[54] & r[53] & r[46] ^ r[54] & r[53] & r[45] ^ r[54] & r[53] ^ r[54] & r[46] & r[45] ^ r[54] & r[46] ^ r[54] ^ r[53] & r[46] & r[45] ^ r[53] ^ r[46] & r[45] ^ r[45];
                bDerivByte12[2] = r[53] & r[52] & r[44] ^ r[53] & r[45] ^ r[52] & r[45] & r[44] ^ r[52] & r[44] ^ r[52] ^ r[44];
                bDerivByte12[1] = r[52] & r[51] & r[43] ^ r[52] & r[51] ^ r[52] & r[44] ^ r[52] & r[43] ^ r[51] & r[44] & r[43] ^ r[51] & r[44] ^ r[51] ^ r[44] & r[43] ^ r[44] ^ r[43];
                bDerivByte12[0] = r[51] & r[50] & r[42] ^ r[51] & r[43] & r[42] ^ r[51] & r[43] ^ r[50] & r[43] & r[42] ^ r[50] ^ r[43] ^ r[42] ^ 0xff;

//                uint8_t bDerivByte12 = gpu_tea3_state_word_to_newbyte((qwIvReg >> 8) & 0xffff, gpu_awTea3LutA);  // F32
//                uint8_t bDerivByte56 = gpu_tea3_state_word_to_newbyte((qwIvReg >> 40) & 0xffff, gpu_awTea3LutB); // F31
/*                bDerivByte56[7] = r[13] & r[14] & r[21] ^ r[13] & r[14] ^ r[13] & r[21] & r[22] ^ r[13] & r[21] ^ r[13] & r[22] ^ r[14] & r[21] & r[22] ^ r[21] & r[22] ^ r[21] ^ r[22];
                bDerivByte56[6] = r[14] & r[15] & r[22] ^ r[14] & r[15] ^ r[14] & r[22] & r[23] ^ r[14] & r[23] ^ r[14] ^ r[15] & r[22] ^ r[15] & r[23] ^ r[22] & r[23] ^ r[22] ^ r[23] ^ 0xff;
                bDerivByte56[5] = r[8] & r[15] & r[16] ^ r[8] & r[16] & r[23] ^ r[8] & r[16] ^ r[8] ^ r[15] & r[23] ^ r[16];
                bDerivByte56[4] = r[8] & r[9] & r[16] ^ r[8] & r[9] & r[17] ^ r[8] & r[9] ^ r[8] & r[16] & r[17] ^ r[8] & r[17] ^ r[9] & r[16] & r[17] ^ r[9] ^ r[17];
                bDerivByte56[3] = r[9] & r[10] & r[18] ^ r[9] & r[10] ^ r[9] & r[17] ^ r[9] & r[18] ^ r[10] & r[17] & r[18] ^ r[10] & r[17] ^ r[10] ^ r[17] & r[18] ^ r[17] ^ r[18] ^ 0xff;
                bDerivByte56[2] = r[10] & r[11] & r[19] ^ r[10] & r[18] ^ r[10] ^ r[11] & r[18] & r[19] ^ r[11] & r[19] ^ r[11] ^ r[18] ^ r[19] ^ 0xff;
                bDerivByte56[1] = r[11] & r[12] & r[20] ^ r[11] & r[19] ^ r[11] & r[20] ^ r[12] & r[19] & r[20] ^ r[12] & r[20] ^ r[12] ^ r[19] & r[20] ^ 0xff;
                bDerivByte56[0] = r[12] & r[13] & r[21] ^ r[12] & r[20] & r[21] ^ r[12] & r[20] ^ r[12] & r[21] ^ r[13] & r[20] & r[21] ^ r[13] ^ r[21] ^ 0xff;*/ //old


                bDerivByte56[7] = r[18] & r[17] & r[10] ^ r[18] & r[17] ^ r[18] & r[10] & r[9] ^ r[18] & r[10] ^ r[18] & r[9] ^ r[17] & r[10] & r[9] ^ r[10] & r[9] ^ r[10] ^ r[9];
                bDerivByte56[6] = r[17] & r[16] & r[9] ^ r[17] & r[16] ^ r[17] & r[9] & r[8] ^ r[17] & r[8] ^ r[17] ^ r[16] & r[9] ^ r[16] & r[8] ^ r[9] & r[8] ^ r[9] ^ r[8] ^ 0xff;
                bDerivByte56[5] = r[23] & r[16] & r[15] ^ r[23] & r[15] & r[8] ^ r[23] & r[15] ^ r[23] ^ r[16] & r[8] ^ r[15];
                bDerivByte56[4] = r[23] & r[22] & r[15] ^ r[23] & r[22] & r[14] ^ r[23] & r[22] ^ r[23] & r[15] & r[14] ^ r[23] & r[14] ^ r[22] & r[15] & r[14] ^ r[22] ^ r[14];
                bDerivByte56[3] = r[22] & r[21] & r[13] ^ r[22] & r[21] ^ r[22] & r[14] ^ r[22] & r[13] ^ r[21] & r[14] & r[13] ^ r[21] & r[14] ^ r[21] ^ r[14] & r[13] ^ r[14] ^ r[13] ^ 0xff;
                bDerivByte56[2] = r[21] & r[20] & r[12] ^ r[21] & r[13] ^ r[21] ^ r[20] & r[13] & r[12] ^ r[20] & r[12] ^ r[20] ^ r[13] ^ r[12] ^ 0xff;
                bDerivByte56[1] = r[20] & r[19] & r[11] ^ r[20] & r[12] ^ r[20] & r[11] ^ r[19] & r[12] & r[11] ^ r[19] & r[11] ^ r[19] ^ r[12] & r[11] ^ 0xff;
                bDerivByte56[0] = r[19] & r[18] & r[10] ^ r[19] & r[11] & r[10] ^ r[19] & r[11] ^ r[19] & r[10] ^ r[18] & r[11] & r[10] ^ r[18] ^ r[10] ^ 0xff;


//                uint8_t bReordByte4 = gpu_tea3_reorder_state_byte((qwIvReg >> 32) & 0xff);
                bReordByte4[0] = r[26];
                bReordByte4[1] = r[31];
                bReordByte4[2] = r[27];
                bReordByte4[3] = r[29];
                bReordByte4[4] = r[30];
                bReordByte4[5] = r[25];
                bReordByte4[6] = r[24];
                bReordByte4[7] = r[28];
 //               if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bDerivByte12[i]); printf("\n");}
  //              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bDerivByte56[i]); printf("\n"); }
  //              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bReordByte4[i]); printf("\n"); }
                // Transpose the S-box output
                int k;
                unsigned m, t;
                m = 0x0000000F;

                for (int l = 4; l != 0; l = l >> 1, m = m ^ (m << l)) {
                    for (k = 0; k < 8; k = (k + l + 1) & ~l) {
                        t = (bSboxOut[k] ^ (bSboxOut[k + l] >> l)) & m;
                        bSboxOut[k] = bSboxOut[k] ^ t;
                        bSboxOut[k + l] = bSboxOut[k + l] ^ (t << l);
                    }
                }

                // Transpose the S-box (my implementation)
 /*               uint8_t bSboxOut2[8] = { 0 };
                for (int i = 0; i < 8; i++) {
                    for (int j = 7; j > 0; j--) {
                        bSboxOut2[i] |= bSboxOut[j] & (0x80 >> i);
                        bSboxOut2[i] >>= 1;
                    }
                    bSboxOut2[i] |= bSboxOut[0] & (0x80 >> i);
                }*/

 //               if (threadIndex == 0) for (int i = 0; i < 8; i++) printf("%02x\n", bSboxOut[i]); // Sboxes are transposed correctly
                // Step 3: Combine current state with state derived values, and xor in key derived sbox output
//                uint8_t bNewByte = ((qwIvReg >> 56) ^ bReordByte4 ^ bDerivByte12 ^ bSboxOut) & 0xff;
/*                bNewByte[0] = r[0] ^ bReordByte4[0] ^ bDerivByte12[0] ^ bSboxOut[0]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[1] = r[1] ^ bReordByte4[1] ^ bDerivByte12[1] ^ bSboxOut[1]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[2] = r[2] ^ bReordByte4[2] ^ bDerivByte12[2] ^ bSboxOut[2]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[3] = r[3] ^ bReordByte4[3] ^ bDerivByte12[3] ^ bSboxOut[3]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[4] = r[4] ^ bReordByte4[4] ^ bDerivByte12[4] ^ bSboxOut[4]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[5] = r[5] ^ bReordByte4[5] ^ bDerivByte12[5] ^ bSboxOut[5]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[6] = r[6] ^ bReordByte4[6] ^ bDerivByte12[6] ^ bSboxOut[6]; // bSboxOut must be turned into bSboxOut[8]
                bNewByte[7] = r[7] ^ bReordByte4[7] ^ bDerivByte12[7] ^ bSboxOut[7]; // bSboxOut must be turned into bSboxOut[8]
*/
                for (int i = 0; i < 8; i++) bNewByte[i] = r[i] ^ bReordByte4[i] ^ bDerivByte12[i] ^ bSboxOut[i];



 //               uint8_t bMixByte = bDerivByte56;
                // Step 4: Update lfsr: leftshift 8, feed/mix in previously generated bytes
 //               qwIvReg = ((qwIvReg << 8) ^ ((uint64_t)bMixByte << 40)) | bNewByte;

                r[0] = r[8];
                r[1] = r[9];
                r[2] = r[10];
                r[3] = r[11];
                r[4] = r[12];
                r[5] = r[13];
                r[6] = r[14];
                r[7] = r[15];
                r[8] = r[16];
                r[9] = r[17];
                r[10] = r[18];
                r[11] = r[19];
                r[12] = r[20];
                r[13] = r[21];
                r[14] = r[22];
                r[15] = r[23];
                r[16] = r[24] ^ bDerivByte56[0];
                r[17] = r[25] ^ bDerivByte56[1];
                r[18] = r[26] ^ bDerivByte56[2];
                r[19] = r[27] ^ bDerivByte56[3];
                r[20] = r[28] ^ bDerivByte56[4];
                r[21] = r[29] ^ bDerivByte56[5];
                r[22] = r[30] ^ bDerivByte56[6];
                r[23] = r[31] ^ bDerivByte56[7];
                r[24] = r[32];
                r[25] = r[33];
                r[26] = r[34];
                r[27] = r[35];
                r[28] = r[36];
                r[29] = r[37];
                r[30] = r[38];
                r[31] = r[39];
                r[32] = r[40];
                r[33] = r[41];
                r[34] = r[42];
                r[35] = r[43];
                r[36] = r[44];
                r[37] = r[45];
                r[38] = r[46];
                r[39] = r[47];
                r[40] = r[48];
                r[41] = r[49];
                r[42] = r[50];
                r[43] = r[51];
                r[44] = r[52];
                r[45] = r[53];
                r[46] = r[54];
                r[47] = r[55];
                r[48] = r[56];
                r[49] = r[57];
                r[50] = r[58];
                r[51] = r[59];
                r[52] = r[60];
                r[53] = r[61];
                r[54] = r[62];
                r[55] = r[63];

                r[56] = bNewByte[0];
                r[57] = bNewByte[1];
                r[58] = bNewByte[2];
                r[59] = bNewByte[3];
                r[60] = bNewByte[4];
                r[61] = bNewByte[5];
                r[62] = bNewByte[6];
                r[63] = bNewByte[7];

            }
            //           lpKsOut[i] = (qwIvReg >> 56);
 //           if ((qwIvReg >> 56) != lpKsOut[i]) flag = 0;
            for (int i = 0; i < 8; i++)
                for (int c = 0; c < 8; c++)
                    if ((r[c] & (0x80>>i)) != (lpKsOut[w*8+c] & (0x80 >> i))) flag[i] = 0;
            //            if (threadIndex == 0 && trial==0) printf("%llx %x\n", qwIvReg >> 56, lpKsOut[i]);
            dwNumSkipRounds = 19;
            flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7];
        }
//        if (threadIndex == 0) for (int i = 0; i < 8;i++) printf("%x\n", r[i]);
        if (flag_overall == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial; }
        //        if (flag == 1) printf("Hello world %llu\n",threadIndex);

    }
}
__global__ void tea3_exhaustive_bitsliced_shared(uint8_t* reg, uint32_t dwNumKsBytes, uint8_t* lpKsOut_d, uint8_t* gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint32_t gpu_abTea3Sbox[256][32];
    __shared__ uint8_t lpKsOut[80];
    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 256) {
        for (int i = 0; i < 32; i++) gpu_abTea3Sbox[threadIdx.x][i] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 80) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
    }
    __syncthreads();
    uint16_t key_right[8]; // rightmost 16 bits
    uint64_t key_left[8]; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    uint8_t r[64];
    uint8_t bDerivByte12[8], bDerivByte56[8], bReordByte4[8], bNewByte[8];
    int flag[8];
    //    if (threadIndex == 0) for (int i = 0; i < 3; i++) printf("Reg %d: %x\n", i, reg[i]);
        //   threadIndex = threadIndex << (64 - BLOCKSLOG - THREADSLOG);    
    for (int trial = 0; trial < trials; trial++) {
        uint32_t dwNumSkipRounds = 51;
        for (int i = 0; i < 64; i++) r[i] = reg[i];
        for (uint64_t i = 0; i < 8; i++) {
            key_right[i] = trial; // rightmost 16 bits
            key_left[i] = (threadIndex << (64 - BLOCKSLOG - THREADSLOG)) ^ (i << 36); // leftmost 64 bits
            flag[i] = 1;
        }
        int flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7];
        for (int w = 0; (w < dwNumKsBytes) && flag_overall == 1; w++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint8_t bSboxOut[8];
                for (int i = 0; i < 8; i++) {
                    bSboxOut[i] = gpu_abTea3Sbox[(key_left[i] & 0xff) ^ ((key_left[i] >> 40) & 0xff)][warpThreadIndex] ^ (key_left[i] >> 56);
                    key_left[i] = (key_left[i] << 8) ^ (key_right[i] >> 8);
                    key_right[i] = (key_right[i] << 8) ^ bSboxOut[i];
                }
                //               if (threadIndex == 0) for (int i = 0; i < 8; i++) printf("%llx %x\n",key_left[i], key_right[i]); // keys are generated correctly 
                //               if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 64; i++) printf("%02x\n", reg[i]); printf("\n"); }
                               // Step 2: Compute 3 bytes derived from current state
               /*                bDerivByte12[7] = r[45] & r[46] & r[53] ^ r[45] & r[46] ^ r[45] & r[53] & r[54] ^ r[45] & r[53] ^ r[45] & r[54] ^ r[46] & r[53] & r[54] ^ r[53] ^ r[54] ^ 0xff;
                               bDerivByte12[6] = r[46] & r[47] & r[54] ^ r[46] & r[47] ^ r[46] & r[54] & r[55] ^ r[46] & r[55] ^ r[46] ^ r[47] & r[55] ^ r[47] ^ r[54] ^ 0xff;
                               bDerivByte12[5] = r[40] & r[47] & r[48] ^ r[40] & r[47] & r[55] ^ r[40] & r[48] & r[55] ^ r[40] & r[48] ^ r[40] ^ r[47] & r[48] & r[55] ^ r[48];
                               bDerivByte12[4] = r[40] & r[41] & r[48] ^ r[40] & r[41] & r[49] ^ r[40] & r[48] & r[49] ^ r[41] & r[48] & r[49] ^ r[41] & r[48] ^ r[41] ^ r[48] & r[49] ^ r[49];
                               bDerivByte12[3] = r[41] & r[42] & r[49] ^ r[41] & r[42] & r[50] ^ r[41] & r[42] ^ r[41] & r[49] & r[50] ^ r[41] & r[49] ^ r[41] ^ r[42] & r[49] & r[50] ^ r[42] ^ r[49] & r[50] ^ r[50];
                               bDerivByte12[2] = r[42] & r[43] & r[51] ^ r[42] & r[50] ^ r[43] & r[50] & r[51] ^ r[43] & r[51] ^ r[43] ^ r[51];
                               bDerivByte12[1] = r[43] & r[44] & r[52] ^ r[43] & r[44] ^ r[43] & r[51] ^ r[43] & r[52] ^ r[44] & r[51] & r[52] ^ r[44] & r[51] ^ r[44] ^ r[51] & r[52] ^ r[51] ^ r[52];
                               bDerivByte12[0] = r[44] & r[45] & r[53] ^ r[44] & r[52] & r[53] ^ r[44] & r[52] ^ r[45] & r[52] & r[53] ^ r[45] ^ r[52] ^ r[53] ^ 0xff;*/ //old

                bDerivByte12[7] = r[50] & r[49] & r[42] ^ r[50] & r[49] ^ r[50] & r[42] & r[41] ^ r[50] & r[42] ^ r[50] & r[41] ^ r[49] & r[42] & r[41] ^ r[42] ^ r[41] ^ 0xff;
                bDerivByte12[6] = r[49] & r[48] & r[41] ^ r[49] & r[48] ^ r[49] & r[41] & r[40] ^ r[49] & r[40] ^ r[49] ^ r[48] & r[40] ^ r[48] ^ r[41] ^ 0xff;
                bDerivByte12[5] = r[55] & r[48] & r[47] ^ r[55] & r[48] & r[40] ^ r[55] & r[47] & r[40] ^ r[55] & r[47] ^ r[55] ^ r[48] & r[47] & r[40] ^ r[47];
                bDerivByte12[4] = r[55] & r[54] & r[47] ^ r[55] & r[54] & r[46] ^ r[55] & r[47] & r[46] ^ r[54] & r[47] & r[46] ^ r[54] & r[47] ^ r[54] ^ r[47] & r[46] ^ r[46];
                bDerivByte12[3] = r[54] & r[53] & r[46] ^ r[54] & r[53] & r[45] ^ r[54] & r[53] ^ r[54] & r[46] & r[45] ^ r[54] & r[46] ^ r[54] ^ r[53] & r[46] & r[45] ^ r[53] ^ r[46] & r[45] ^ r[45];
                bDerivByte12[2] = r[53] & r[52] & r[44] ^ r[53] & r[45] ^ r[52] & r[45] & r[44] ^ r[52] & r[44] ^ r[52] ^ r[44];
                bDerivByte12[1] = r[52] & r[51] & r[43] ^ r[52] & r[51] ^ r[52] & r[44] ^ r[52] & r[43] ^ r[51] & r[44] & r[43] ^ r[51] & r[44] ^ r[51] ^ r[44] & r[43] ^ r[44] ^ r[43];
                bDerivByte12[0] = r[51] & r[50] & r[42] ^ r[51] & r[43] & r[42] ^ r[51] & r[43] ^ r[50] & r[43] & r[42] ^ r[50] ^ r[43] ^ r[42] ^ 0xff;

                //                uint8_t bDerivByte12 = gpu_tea3_state_word_to_newbyte((qwIvReg >> 8) & 0xffff, gpu_awTea3LutA);  // F32
                //                uint8_t bDerivByte56 = gpu_tea3_state_word_to_newbyte((qwIvReg >> 40) & 0xffff, gpu_awTea3LutB); // F31
                /*                bDerivByte56[7] = r[13] & r[14] & r[21] ^ r[13] & r[14] ^ r[13] & r[21] & r[22] ^ r[13] & r[21] ^ r[13] & r[22] ^ r[14] & r[21] & r[22] ^ r[21] & r[22] ^ r[21] ^ r[22];
                                bDerivByte56[6] = r[14] & r[15] & r[22] ^ r[14] & r[15] ^ r[14] & r[22] & r[23] ^ r[14] & r[23] ^ r[14] ^ r[15] & r[22] ^ r[15] & r[23] ^ r[22] & r[23] ^ r[22] ^ r[23] ^ 0xff;
                                bDerivByte56[5] = r[8] & r[15] & r[16] ^ r[8] & r[16] & r[23] ^ r[8] & r[16] ^ r[8] ^ r[15] & r[23] ^ r[16];
                                bDerivByte56[4] = r[8] & r[9] & r[16] ^ r[8] & r[9] & r[17] ^ r[8] & r[9] ^ r[8] & r[16] & r[17] ^ r[8] & r[17] ^ r[9] & r[16] & r[17] ^ r[9] ^ r[17];
                                bDerivByte56[3] = r[9] & r[10] & r[18] ^ r[9] & r[10] ^ r[9] & r[17] ^ r[9] & r[18] ^ r[10] & r[17] & r[18] ^ r[10] & r[17] ^ r[10] ^ r[17] & r[18] ^ r[17] ^ r[18] ^ 0xff;
                                bDerivByte56[2] = r[10] & r[11] & r[19] ^ r[10] & r[18] ^ r[10] ^ r[11] & r[18] & r[19] ^ r[11] & r[19] ^ r[11] ^ r[18] ^ r[19] ^ 0xff;
                                bDerivByte56[1] = r[11] & r[12] & r[20] ^ r[11] & r[19] ^ r[11] & r[20] ^ r[12] & r[19] & r[20] ^ r[12] & r[20] ^ r[12] ^ r[19] & r[20] ^ 0xff;
                                bDerivByte56[0] = r[12] & r[13] & r[21] ^ r[12] & r[20] & r[21] ^ r[12] & r[20] ^ r[12] & r[21] ^ r[13] & r[20] & r[21] ^ r[13] ^ r[21] ^ 0xff;*/ //old


                bDerivByte56[7] = r[18] & r[17] & r[10] ^ r[18] & r[17] ^ r[18] & r[10] & r[9] ^ r[18] & r[10] ^ r[18] & r[9] ^ r[17] & r[10] & r[9] ^ r[10] & r[9] ^ r[10] ^ r[9];
                bDerivByte56[6] = r[17] & r[16] & r[9] ^ r[17] & r[16] ^ r[17] & r[9] & r[8] ^ r[17] & r[8] ^ r[17] ^ r[16] & r[9] ^ r[16] & r[8] ^ r[9] & r[8] ^ r[9] ^ r[8] ^ 0xff;
                bDerivByte56[5] = r[23] & r[16] & r[15] ^ r[23] & r[15] & r[8] ^ r[23] & r[15] ^ r[23] ^ r[16] & r[8] ^ r[15];
                bDerivByte56[4] = r[23] & r[22] & r[15] ^ r[23] & r[22] & r[14] ^ r[23] & r[22] ^ r[23] & r[15] & r[14] ^ r[23] & r[14] ^ r[22] & r[15] & r[14] ^ r[22] ^ r[14];
                bDerivByte56[3] = r[22] & r[21] & r[13] ^ r[22] & r[21] ^ r[22] & r[14] ^ r[22] & r[13] ^ r[21] & r[14] & r[13] ^ r[21] & r[14] ^ r[21] ^ r[14] & r[13] ^ r[14] ^ r[13] ^ 0xff;
                bDerivByte56[2] = r[21] & r[20] & r[12] ^ r[21] & r[13] ^ r[21] ^ r[20] & r[13] & r[12] ^ r[20] & r[12] ^ r[20] ^ r[13] ^ r[12] ^ 0xff;
                bDerivByte56[1] = r[20] & r[19] & r[11] ^ r[20] & r[12] ^ r[20] & r[11] ^ r[19] & r[12] & r[11] ^ r[19] & r[11] ^ r[19] ^ r[12] & r[11] ^ 0xff;
                bDerivByte56[0] = r[19] & r[18] & r[10] ^ r[19] & r[11] & r[10] ^ r[19] & r[11] ^ r[19] & r[10] ^ r[18] & r[11] & r[10] ^ r[18] ^ r[10] ^ 0xff;


                //                uint8_t bReordByte4 = gpu_tea3_reorder_state_byte((qwIvReg >> 32) & 0xff);
                bReordByte4[0] = r[26];
                bReordByte4[1] = r[31];
                bReordByte4[2] = r[27];
                bReordByte4[3] = r[29];
                bReordByte4[4] = r[30];
                bReordByte4[5] = r[25];
                bReordByte4[6] = r[24];
                bReordByte4[7] = r[28];
                //               if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bDerivByte12[i]); printf("\n");}
                 //              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bDerivByte56[i]); printf("\n"); }
                 //              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bReordByte4[i]); printf("\n"); }
                               // Transpose the S-box output
                int k;
                unsigned m, t;
                m = 0x0000000F;

                for (int l = 4; l != 0; l = l >> 1, m = m ^ (m << l)) {
                    for (k = 0; k < 8; k = (k + l + 1) & ~l) {
                        t = (bSboxOut[k] ^ (bSboxOut[k + l] >> l)) & m;
                        bSboxOut[k] = bSboxOut[k] ^ t;
                        bSboxOut[k + l] = bSboxOut[k + l] ^ (t << l);
                    }
                }

                // Transpose the S-box (my implementation)
 /*               uint8_t bSboxOut2[8] = { 0 };
                for (int i = 0; i < 8; i++) {
                    for (int j = 7; j > 0; j--) {
                        bSboxOut2[i] |= bSboxOut[j] & (0x80 >> i);
                        bSboxOut2[i] >>= 1;
                    }
                    bSboxOut2[i] |= bSboxOut[0] & (0x80 >> i);
                }*/

                //               if (threadIndex == 0) for (int i = 0; i < 8; i++) printf("%02x\n", bSboxOut[i]); // Sboxes are transposed correctly
                               // Step 3: Combine current state with state derived values, and xor in key derived sbox output
               //                uint8_t bNewByte = ((qwIvReg >> 56) ^ bReordByte4 ^ bDerivByte12 ^ bSboxOut) & 0xff;
               /*                bNewByte[0] = r[0] ^ bReordByte4[0] ^ bDerivByte12[0] ^ bSboxOut[0]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[1] = r[1] ^ bReordByte4[1] ^ bDerivByte12[1] ^ bSboxOut[1]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[2] = r[2] ^ bReordByte4[2] ^ bDerivByte12[2] ^ bSboxOut[2]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[3] = r[3] ^ bReordByte4[3] ^ bDerivByte12[3] ^ bSboxOut[3]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[4] = r[4] ^ bReordByte4[4] ^ bDerivByte12[4] ^ bSboxOut[4]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[5] = r[5] ^ bReordByte4[5] ^ bDerivByte12[5] ^ bSboxOut[5]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[6] = r[6] ^ bReordByte4[6] ^ bDerivByte12[6] ^ bSboxOut[6]; // bSboxOut must be turned into bSboxOut[8]
                               bNewByte[7] = r[7] ^ bReordByte4[7] ^ bDerivByte12[7] ^ bSboxOut[7]; // bSboxOut must be turned into bSboxOut[8]
               */
                for (int i = 0; i < 8; i++) bNewByte[i] = r[i] ^ bReordByte4[i] ^ bDerivByte12[i] ^ bSboxOut[i];



                //               uint8_t bMixByte = bDerivByte56;
                               // Step 4: Update lfsr: leftshift 8, feed/mix in previously generated bytes
                //               qwIvReg = ((qwIvReg << 8) ^ ((uint64_t)bMixByte << 40)) | bNewByte;

                r[0] = r[8];
                r[1] = r[9];
                r[2] = r[10];
                r[3] = r[11];
                r[4] = r[12];
                r[5] = r[13];
                r[6] = r[14];
                r[7] = r[15];
                r[8] = r[16];
                r[9] = r[17];
                r[10] = r[18];
                r[11] = r[19];
                r[12] = r[20];
                r[13] = r[21];
                r[14] = r[22];
                r[15] = r[23];
                r[16] = r[24] ^ bDerivByte56[0];
                r[17] = r[25] ^ bDerivByte56[1];
                r[18] = r[26] ^ bDerivByte56[2];
                r[19] = r[27] ^ bDerivByte56[3];
                r[20] = r[28] ^ bDerivByte56[4];
                r[21] = r[29] ^ bDerivByte56[5];
                r[22] = r[30] ^ bDerivByte56[6];
                r[23] = r[31] ^ bDerivByte56[7];
                r[24] = r[32];
                r[25] = r[33];
                r[26] = r[34];
                r[27] = r[35];
                r[28] = r[36];
                r[29] = r[37];
                r[30] = r[38];
                r[31] = r[39];
                r[32] = r[40];
                r[33] = r[41];
                r[34] = r[42];
                r[35] = r[43];
                r[36] = r[44];
                r[37] = r[45];
                r[38] = r[46];
                r[39] = r[47];
                r[40] = r[48];
                r[41] = r[49];
                r[42] = r[50];
                r[43] = r[51];
                r[44] = r[52];
                r[45] = r[53];
                r[46] = r[54];
                r[47] = r[55];
                r[48] = r[56];
                r[49] = r[57];
                r[50] = r[58];
                r[51] = r[59];
                r[52] = r[60];
                r[53] = r[61];
                r[54] = r[62];
                r[55] = r[63];

                r[56] = bNewByte[0];
                r[57] = bNewByte[1];
                r[58] = bNewByte[2];
                r[59] = bNewByte[3];
                r[60] = bNewByte[4];
                r[61] = bNewByte[5];
                r[62] = bNewByte[6];
                r[63] = bNewByte[7];

            }
            //           lpKsOut[i] = (qwIvReg >> 56);
 //           if ((qwIvReg >> 56) != lpKsOut[i]) flag = 0;
            for (int i = 0; i < 8; i++)
                for (int c = 0; c < 8; c++)
                    if ((r[c] & (0x80 >> i)) != (lpKsOut[w * 8 + c] & (0x80 >> i))) flag[i] = 0;
            //            if (threadIndex == 0 && trial==0) printf("%llx %x\n", qwIvReg >> 56, lpKsOut[i]);
            dwNumSkipRounds = 19;
            flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7];
        }
        //        if (threadIndex == 0) for (int i = 0; i < 8;i++) printf("%x\n", r[i]);
        if (flag_overall == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial; }
        //        if (flag == 1) printf("Hello world %llu\n",threadIndex);

    }
}
// tea3_exhaustive_bitsliced16 kernel keeps each bit of the state in a 16-bit variable
// These 16-bit variables keep 16 different states so one encryption performed by a thread actually means 16 encryptions in parallel
__global__ void tea3_exhaustive_bitsliced16(uint16_t* reg, uint32_t dwNumKsBytes, uint16_t* lpKsOut_d, uint8_t* gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint8_t gpu_abTea3Sbox[256];
    __shared__ uint16_t lpKsOut[80];
    if (threadIdx.x < 256) {
        gpu_abTea3Sbox[threadIdx.x] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 80) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
    }
    __syncthreads();
    uint16_t key_right[16]; // rightmost 16 bits
    uint64_t key_left[16]; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    uint16_t r[64];
    uint16_t bDerivByte12[8], bDerivByte56[8], bReordByte4[8], bNewByte[8];
    int flag[16];
    for (int trial = 0; trial < trials; trial++) {
        uint32_t dwNumSkipRounds = 51;
        for (int i = 0; i < 64; i++) r[i] = reg[i];
        for (uint64_t i = 0; i < 16; i++) {
            key_right[i] = trial; // rightmost 16 bits
            key_left[i] = (threadIndex << (64 - BLOCKSLOG - THREADSLOG)) ^ (i << 16); // leftmost 64 bits
            flag[i] = 1;
        }
        int flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7] | flag[8] | flag[9] | flag[10] | flag[11] | flag[12] | flag[13] | flag[14] | flag[15];
        for (int w = 0; (w < dwNumKsBytes) && flag_overall == 1; w++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint16_t bSboxOut[16] = { 0 };
                for (int i = 0; i < 16; i++) {
                    bSboxOut[i] = gpu_abTea3Sbox[(key_left[i] & 0xff) ^ ((key_left[i] >> 40) & 0xff)] ^ (key_left[i] >> 56);
                    key_left[i] = (key_left[i] << 8) ^ (key_right[i] >> 8);
                    key_right[i] = (key_right[i] << 8) ^ (bSboxOut[i]&0xff);
                }
                //              if (threadIndex == 0) for (int i = 0; i < 16; i++) printf("%llx %x\n",key_left[i], key_right[i]); // keys are generated correctly 
                //              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 64; i++) printf("%02x\n", reg[i]); printf("\n"); }
                bDerivByte12[7] = r[50] & r[49] & r[42] ^ r[50] & r[49] ^ r[50] & r[42] & r[41] ^ r[50] & r[42] ^ r[50] & r[41] ^ r[49] & r[42] & r[41] ^ r[42] ^ r[41] ^ 0xffff;
                bDerivByte12[6] = r[49] & r[48] & r[41] ^ r[49] & r[48] ^ r[49] & r[41] & r[40] ^ r[49] & r[40] ^ r[49] ^ r[48] & r[40] ^ r[48] ^ r[41] ^ 0xffff;
                bDerivByte12[5] = r[55] & r[48] & r[47] ^ r[55] & r[48] & r[40] ^ r[55] & r[47] & r[40] ^ r[55] & r[47] ^ r[55] ^ r[48] & r[47] & r[40] ^ r[47];
                bDerivByte12[4] = r[55] & r[54] & r[47] ^ r[55] & r[54] & r[46] ^ r[55] & r[47] & r[46] ^ r[54] & r[47] & r[46] ^ r[54] & r[47] ^ r[54] ^ r[47] & r[46] ^ r[46];
                bDerivByte12[3] = r[54] & r[53] & r[46] ^ r[54] & r[53] & r[45] ^ r[54] & r[53] ^ r[54] & r[46] & r[45] ^ r[54] & r[46] ^ r[54] ^ r[53] & r[46] & r[45] ^ r[53] ^ r[46] & r[45] ^ r[45];
                bDerivByte12[2] = r[53] & r[52] & r[44] ^ r[53] & r[45] ^ r[52] & r[45] & r[44] ^ r[52] & r[44] ^ r[52] ^ r[44];
                bDerivByte12[1] = r[52] & r[51] & r[43] ^ r[52] & r[51] ^ r[52] & r[44] ^ r[52] & r[43] ^ r[51] & r[44] & r[43] ^ r[51] & r[44] ^ r[51] ^ r[44] & r[43] ^ r[44] ^ r[43];
                bDerivByte12[0] = r[51] & r[50] & r[42] ^ r[51] & r[43] & r[42] ^ r[51] & r[43] ^ r[50] & r[43] & r[42] ^ r[50] ^ r[43] ^ r[42] ^ 0xffff;

                bDerivByte56[7] = r[18] & r[17] & r[10] ^ r[18] & r[17] ^ r[18] & r[10] & r[9] ^ r[18] & r[10] ^ r[18] & r[9] ^ r[17] & r[10] & r[9] ^ r[10] & r[9] ^ r[10] ^ r[9];
                bDerivByte56[6] = r[17] & r[16] & r[9] ^ r[17] & r[16] ^ r[17] & r[9] & r[8] ^ r[17] & r[8] ^ r[17] ^ r[16] & r[9] ^ r[16] & r[8] ^ r[9] & r[8] ^ r[9] ^ r[8] ^ 0xffff;
                bDerivByte56[5] = r[23] & r[16] & r[15] ^ r[23] & r[15] & r[8] ^ r[23] & r[15] ^ r[23] ^ r[16] & r[8] ^ r[15];
                bDerivByte56[4] = r[23] & r[22] & r[15] ^ r[23] & r[22] & r[14] ^ r[23] & r[22] ^ r[23] & r[15] & r[14] ^ r[23] & r[14] ^ r[22] & r[15] & r[14] ^ r[22] ^ r[14];
                bDerivByte56[3] = r[22] & r[21] & r[13] ^ r[22] & r[21] ^ r[22] & r[14] ^ r[22] & r[13] ^ r[21] & r[14] & r[13] ^ r[21] & r[14] ^ r[21] ^ r[14] & r[13] ^ r[14] ^ r[13] ^ 0xffff;
                bDerivByte56[2] = r[21] & r[20] & r[12] ^ r[21] & r[13] ^ r[21] ^ r[20] & r[13] & r[12] ^ r[20] & r[12] ^ r[20] ^ r[13] ^ r[12] ^ 0xffff;
                bDerivByte56[1] = r[20] & r[19] & r[11] ^ r[20] & r[12] ^ r[20] & r[11] ^ r[19] & r[12] & r[11] ^ r[19] & r[11] ^ r[19] ^ r[12] & r[11] ^ 0xffff;
                bDerivByte56[0] = r[19] & r[18] & r[10] ^ r[19] & r[11] & r[10] ^ r[19] & r[11] ^ r[19] & r[10] ^ r[18] & r[11] & r[10] ^ r[18] ^ r[10] ^ 0xffff;

                bReordByte4[0] = r[26];
                bReordByte4[1] = r[31];
                bReordByte4[2] = r[27];
                bReordByte4[3] = r[29];
                bReordByte4[4] = r[30];
                bReordByte4[5] = r[25];
                bReordByte4[6] = r[24];
                bReordByte4[7] = r[28];

//                              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bDerivByte12[i]); printf("\n");}
//              if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bDerivByte56[i]); printf("\n"); }
//               if (threadIndex == 0) { printf("\n"); for (int i = 0; i < 8; i++) printf("%x\n", bReordByte4[i]); printf("\n"); }
                               // Transpose the S-box output
                int k;
                unsigned m, t;
                m = 0x000000FF;

                for (int l = 8; l != 0; l = l >> 1, m = m ^ (m << l)) {
                    for (k = 0; k < 16; k = (k + l + 1) & ~l) {
                        t = (bSboxOut[k] ^ (bSboxOut[k + l] >> l)) & m;
                        bSboxOut[k] = bSboxOut[k] ^ t;
                        bSboxOut[k + l] = bSboxOut[k + l] ^ (t << l);
                    }
                }

                for (int i = 0; i < 8; i++) bNewByte[i] = r[i] ^ bReordByte4[i] ^ bDerivByte12[i] ^ bSboxOut[i+8];

                r[0] = r[8];
                r[1] = r[9];
                r[2] = r[10];
                r[3] = r[11];
                r[4] = r[12];
                r[5] = r[13];
                r[6] = r[14];
                r[7] = r[15];
                r[8] = r[16];
                r[9] = r[17];
                r[10] = r[18];
                r[11] = r[19];
                r[12] = r[20];
                r[13] = r[21];
                r[14] = r[22];
                r[15] = r[23];
                r[16] = r[24] ^ bDerivByte56[0];
                r[17] = r[25] ^ bDerivByte56[1];
                r[18] = r[26] ^ bDerivByte56[2];
                r[19] = r[27] ^ bDerivByte56[3];
                r[20] = r[28] ^ bDerivByte56[4];
                r[21] = r[29] ^ bDerivByte56[5];
                r[22] = r[30] ^ bDerivByte56[6];
                r[23] = r[31] ^ bDerivByte56[7];
                r[24] = r[32];
                r[25] = r[33];
                r[26] = r[34];
                r[27] = r[35];
                r[28] = r[36];
                r[29] = r[37];
                r[30] = r[38];
                r[31] = r[39];
                r[32] = r[40];
                r[33] = r[41];
                r[34] = r[42];
                r[35] = r[43];
                r[36] = r[44];
                r[37] = r[45];
                r[38] = r[46];
                r[39] = r[47];
                r[40] = r[48];
                r[41] = r[49];
                r[42] = r[50];
                r[43] = r[51];
                r[44] = r[52];
                r[45] = r[53];
                r[46] = r[54];
                r[47] = r[55];
                r[48] = r[56];
                r[49] = r[57];
                r[50] = r[58];
                r[51] = r[59];
                r[52] = r[60];
                r[53] = r[61];
                r[54] = r[62];
                r[55] = r[63];

                r[56] = bNewByte[0];
                r[57] = bNewByte[1];
                r[58] = bNewByte[2];
                r[59] = bNewByte[3];
                r[60] = bNewByte[4];
                r[61] = bNewByte[5];
                r[62] = bNewByte[6];
                r[63] = bNewByte[7];

            }
            for (int i = 0; i < 16; i++)
                for (int c = 0; c < 8; c++)
                    if ((r[c] & (0x8000 >> i)) != (lpKsOut[w * 8 + c] & (0x8000 >> i))) flag[i] = 0;
            dwNumSkipRounds = 19;
            flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7] | flag[8] | flag[9] | flag[10] | flag[11] | flag[12] | flag[13] | flag[14] | flag[15];
        }
        if (flag_overall == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial; }
    }
}
// tea3_exhaustive_bitsliced32 kernel keeps each bit of the state in a 32-bit variable
// These 32-bit variables keep 32 different states so one encryption performed by a thread actually means 32 encryptions in parallel
__global__ void tea3_exhaustive_bitsliced32(uint32_t* reg, uint32_t dwNumKsBytes, uint32_t* lpKsOut_d, uint8_t* gpu_abTea3Sbox_d, int trials, uint64_t* gpu_captured_key) {
    __shared__ uint8_t gpu_abTea3Sbox[256];
    __shared__ uint32_t lpKsOut[80];
    if (threadIdx.x < 256) {
        gpu_abTea3Sbox[threadIdx.x] = gpu_abTea3Sbox_d[threadIdx.x];
        if (threadIdx.x < 80) lpKsOut[threadIdx.x] = lpKsOut_d[threadIdx.x];
    }
    __syncthreads();
    uint16_t key_right[16]; // rightmost 16 bits
    uint64_t key_left[16]; // leftmost 64 bits
    uint64_t threadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t r[64];
    uint32_t bDerivByte12[8], bDerivByte56[8], bReordByte4[8], bNewByte[8];
    int flag[32];
    for (int trial = 0; trial < trials; trial++) {
        uint32_t dwNumSkipRounds = 51;
#pragma unroll
        for (int i = 0; i < 64; i++) r[i] = reg[i];
#pragma unroll
        for (uint64_t i = 0; i < 32; i++) {
            key_right[i] = trial; // rightmost 16 bits
            key_left[i] = threadIndex ^ (i << 36); // leftmost 64 bits
            flag[i] = 1;
        }
        int flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7] | flag[8] | flag[9] | flag[10] | flag[11] | flag[12] | flag[13] | flag[14] | flag[15]
            | flag[16] | flag[17] | flag[18] | flag[19] | flag[20] | flag[21] | flag[22] | flag[23] | flag[24] | flag[25] | flag[26] | flag[27] | flag[28] | flag[29] | flag[30] | flag[31];
        for (int w = 0; (w < dwNumKsBytes) && flag_overall == 1; w++) {
            for (int j = 0; j < dwNumSkipRounds; j++) {
                // Step 1: Derive a non-linear feedback byte through sbox and feed back into key register
                uint32_t bSboxOut[32] = { 0 };
#pragma unroll
                for (int i = 0; i < 32; i++) {
                    bSboxOut[i] = gpu_abTea3Sbox[(key_left[i] & 0xff) ^ ((key_left[i] >> 40) & 0xff)] ^ (key_left[i] >> 56);
                    key_left[i] = (key_left[i] << 8) ^ (key_right[i] >> 8);
                    key_right[i] = (key_right[i] << 8) ^ (bSboxOut[i] & 0xff);
                }
                bDerivByte12[7] = r[50] & r[49] & r[42] ^ r[50] & r[49] ^ r[50] & r[42] & r[41] ^ r[50] & r[42] ^ r[50] & r[41] ^ r[49] & r[42] & r[41] ^ r[42] ^ r[41] ^ 0xffffffff;
                bDerivByte12[6] = r[49] & r[48] & r[41] ^ r[49] & r[48] ^ r[49] & r[41] & r[40] ^ r[49] & r[40] ^ r[49] ^ r[48] & r[40] ^ r[48] ^ r[41] ^ 0xffffffff;
                bDerivByte12[5] = r[55] & r[48] & r[47] ^ r[55] & r[48] & r[40] ^ r[55] & r[47] & r[40] ^ r[55] & r[47] ^ r[55] ^ r[48] & r[47] & r[40] ^ r[47];
                bDerivByte12[4] = r[55] & r[54] & r[47] ^ r[55] & r[54] & r[46] ^ r[55] & r[47] & r[46] ^ r[54] & r[47] & r[46] ^ r[54] & r[47] ^ r[54] ^ r[47] & r[46] ^ r[46];
                bDerivByte12[3] = r[54] & r[53] & r[46] ^ r[54] & r[53] & r[45] ^ r[54] & r[53] ^ r[54] & r[46] & r[45] ^ r[54] & r[46] ^ r[54] ^ r[53] & r[46] & r[45] ^ r[53] ^ r[46] & r[45] ^ r[45];
                bDerivByte12[2] = r[53] & r[52] & r[44] ^ r[53] & r[45] ^ r[52] & r[45] & r[44] ^ r[52] & r[44] ^ r[52] ^ r[44];
                bDerivByte12[1] = r[52] & r[51] & r[43] ^ r[52] & r[51] ^ r[52] & r[44] ^ r[52] & r[43] ^ r[51] & r[44] & r[43] ^ r[51] & r[44] ^ r[51] ^ r[44] & r[43] ^ r[44] ^ r[43];
                bDerivByte12[0] = r[51] & r[50] & r[42] ^ r[51] & r[43] & r[42] ^ r[51] & r[43] ^ r[50] & r[43] & r[42] ^ r[50] ^ r[43] ^ r[42] ^ 0xffffffff;

                bDerivByte56[7] = r[18] & r[17] & r[10] ^ r[18] & r[17] ^ r[18] & r[10] & r[9] ^ r[18] & r[10] ^ r[18] & r[9] ^ r[17] & r[10] & r[9] ^ r[10] & r[9] ^ r[10] ^ r[9];
                bDerivByte56[6] = r[17] & r[16] & r[9] ^ r[17] & r[16] ^ r[17] & r[9] & r[8] ^ r[17] & r[8] ^ r[17] ^ r[16] & r[9] ^ r[16] & r[8] ^ r[9] & r[8] ^ r[9] ^ r[8] ^ 0xffffffff;
                bDerivByte56[5] = r[23] & r[16] & r[15] ^ r[23] & r[15] & r[8] ^ r[23] & r[15] ^ r[23] ^ r[16] & r[8] ^ r[15];
                bDerivByte56[4] = r[23] & r[22] & r[15] ^ r[23] & r[22] & r[14] ^ r[23] & r[22] ^ r[23] & r[15] & r[14] ^ r[23] & r[14] ^ r[22] & r[15] & r[14] ^ r[22] ^ r[14];
                bDerivByte56[3] = r[22] & r[21] & r[13] ^ r[22] & r[21] ^ r[22] & r[14] ^ r[22] & r[13] ^ r[21] & r[14] & r[13] ^ r[21] & r[14] ^ r[21] ^ r[14] & r[13] ^ r[14] ^ r[13] ^ 0xffffffff;
                bDerivByte56[2] = r[21] & r[20] & r[12] ^ r[21] & r[13] ^ r[21] ^ r[20] & r[13] & r[12] ^ r[20] & r[12] ^ r[20] ^ r[13] ^ r[12] ^ 0xffffffff;
                bDerivByte56[1] = r[20] & r[19] & r[11] ^ r[20] & r[12] ^ r[20] & r[11] ^ r[19] & r[12] & r[11] ^ r[19] & r[11] ^ r[19] ^ r[12] & r[11] ^ 0xffffffff;
                bDerivByte56[0] = r[19] & r[18] & r[10] ^ r[19] & r[11] & r[10] ^ r[19] & r[11] ^ r[19] & r[10] ^ r[18] & r[11] & r[10] ^ r[18] ^ r[10] ^ 0xffffffff;

                bReordByte4[0] = r[26];
                bReordByte4[1] = r[31];
                bReordByte4[2] = r[27];
                bReordByte4[3] = r[29];
                bReordByte4[4] = r[30];
                bReordByte4[5] = r[25];
                bReordByte4[6] = r[24];
                bReordByte4[7] = r[28];

                // Transpose the S-box output
                int k;
                unsigned m, t;
                m = 0x0000FFFF;
#pragma unroll
                for (int l = 16; l != 0; l = l >> 1, m = m ^ (m << l)) {
#pragma unroll
                    for (k = 0; k < 32; k = (k + l + 1) & ~l) {
                        t = (bSboxOut[k] ^ (bSboxOut[k + l] >> l)) & m;
                        bSboxOut[k] = bSboxOut[k] ^ t;
                        bSboxOut[k + l] = bSboxOut[k + l] ^ (t << l);
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; i++) bNewByte[i] = r[i] ^ bReordByte4[i] ^ bDerivByte12[i] ^ bSboxOut[i + 24];
                r[0] = r[8];
                r[1] = r[9];
                r[2] = r[10];
                r[3] = r[11];
                r[4] = r[12];
                r[5] = r[13];
                r[6] = r[14];
                r[7] = r[15];
                r[8] = r[16];
                r[9] = r[17];
                r[10] = r[18];
                r[11] = r[19];
                r[12] = r[20];
                r[13] = r[21];
                r[14] = r[22];
                r[15] = r[23];
                r[16] = r[24] ^ bDerivByte56[0];
                r[17] = r[25] ^ bDerivByte56[1];
                r[18] = r[26] ^ bDerivByte56[2];
                r[19] = r[27] ^ bDerivByte56[3];
                r[20] = r[28] ^ bDerivByte56[4];
                r[21] = r[29] ^ bDerivByte56[5];
                r[22] = r[30] ^ bDerivByte56[6];
                r[23] = r[31] ^ bDerivByte56[7];
                r[24] = r[32];
                r[25] = r[33];
                r[26] = r[34];
                r[27] = r[35];
                r[28] = r[36];
                r[29] = r[37];
                r[30] = r[38];
                r[31] = r[39];
                r[32] = r[40];
                r[33] = r[41];
                r[34] = r[42];
                r[35] = r[43];
                r[36] = r[44];
                r[37] = r[45];
                r[38] = r[46];
                r[39] = r[47];
                r[40] = r[48];
                r[41] = r[49];
                r[42] = r[50];
                r[43] = r[51];
                r[44] = r[52];
                r[45] = r[53];
                r[46] = r[54];
                r[47] = r[55];
                r[48] = r[56];
                r[49] = r[57];
                r[50] = r[58];
                r[51] = r[59];
                r[52] = r[60];
                r[53] = r[61];
                r[54] = r[62];
                r[55] = r[63];

                r[56] = bNewByte[0];
                r[57] = bNewByte[1];
                r[58] = bNewByte[2];
                r[59] = bNewByte[3];
                r[60] = bNewByte[4];
                r[61] = bNewByte[5];
                r[62] = bNewByte[6];
                r[63] = bNewByte[7];

            }
#pragma unroll
            for (int i = 0; i < 32; i++)
#pragma unroll
                for (int c = 0; c < 8; c++)
                    if ((r[c] & (0x80000000 >> i)) != (lpKsOut[w * 8 + c] & (0x80000000 >> i))) flag[i] = 0;
            dwNumSkipRounds = 19;
            flag_overall = flag[0] | flag[1] | flag[2] | flag[3] | flag[4] | flag[5] | flag[6] | flag[7] | flag[8] | flag[9] | flag[10] | flag[11] | flag[12] | flag[13] | flag[14] | flag[15]
                | flag[16] | flag[17] | flag[18] | flag[19] | flag[20] | flag[21] | flag[22] | flag[23] | flag[24] | flag[25] | flag[26] | flag[27] | flag[28] | flag[29] | flag[30] | flag[31];
        }
        if (flag_overall == 1) { gpu_captured_key[0] = threadIndex; gpu_captured_key[1] = trial; }
    }
}

void user_input() {
    // Default values of BLOCKS=1024, THREADS=256 mean that a kernel creates 2^18 threads.
    // How many encryptions each thread is going to perform is determined by the user as a power of 2
    // Thus, for a non-bitsliced implementation, a user input of 5 means 2^23 encryptions for a kernel
    // For a bitsliced implementation where 32-bit values store 32 different states, 2^18 threads performe 2^23 encryptions
    
    printf("(1) Exhaustive search on 80-bit keystream\n"
        "(2) Exhaustive key search on 80-bit keystream (0 shared memory bank conflicts)\n"
        "(3) Exhaustive key search on 80-bit keystream (some shared memory bank conflicts)\n"
        "(4) Keystream generation of 80 bits\n"
        "(5) Keystream generation of 80 bits (0 shared memory bank conflicts)\n"
        "(6) Keystream generation of 80 bits(some shared memory bank conflicts)\n"
        "...\n"
        "(11) BITSLICED Exhaustive search on 80-bit keystream (Each thread performs 8 encrypitons in parallel)\n"
        "(12) BITSLICED Exhaustive search on 80-bit keystream (Each thread performs 8 encrypitons in parallel) (0 bank conflicts)\n"
        "(13) BITSLICED Exhaustive search on 80-bit keystream (Each thread performs 16 encrypitons in parallel)\n"
        "(14) BITSLICED Exhaustive search on 80-bit keystream (Each thread performs 32 encrypitons in parallel)\n"
        "Choice: "
    );
    scanf_s("%d", &choice);
    if (choice == 11) printf("Trials 2^21 + ");
    else if (choice == 12) printf("Trials 2^21 + ");
    else if (choice == 13) printf("Trials 2^22 + ");
    else if (choice == 14) printf("Trials 2^23 + ");
    else printf("Trials 2^18 + ");
    scanf_s("%d", &trials);
    trials = 1 << trials;
}

int main() {
    cudaSetDevice(0);
    user_input();
    uint16_t *gpu_awTea3LutA;
    uint16_t *gpu_awTea3LutB;
    uint8_t *gpu_abTea3Sbox;
    uint8_t *gpu_lpKsOut;
    uint64_t *gpu_captured_key; uint64_t captured_key[2] = { 0xffffffffffffffff, 0xffffffffffffffff };    
    uint8_t bitsliced_keystream[80] = { 0 }; uint16_t bitsliced_keystream16[80] = { 0 }; uint32_t bitsliced_keystream32[80] = { 0 };
    uint8_t *bitsliced_keystream_d; uint16_t* bitsliced_keystream16_d; uint32_t* bitsliced_keystream32_d;
    uint32_t dwNumKsBytes = 10;
    // Test vectors 1
 //   uint32_t dwFrameNumbers = 0;// 0xffffffff;
 //   uint8_t lpKsOut[10] = { 0x3b, 0x35, 0x44, 0x30, 0xdc, 0x3d, 0x3f, 0xee, 0x76, 0xcf };
    // Test vectors 2
    uint32_t dwFrameNumbers = 0x176C;
    uint8_t lpKsOut[10] = { 0x88, 0x40, 0x58, 0x9f, 0x88, 0x7b, 0x93, 0xa5, 0xac, 0x91 };
    uint64_t qwIvReg = tea3_compute_iv(dwFrameNumbers);
    uint8_t reg[64] = {0};    uint16_t reg16[64] = { 0 }; uint32_t reg32[64] = { 0 };
    uint8_t *reg_d; uint16_t* reg16_d; uint32_t* reg32_d;
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 8; j++) {
            reg[i] = reg[i] << 1;
            reg[i] ^= (qwIvReg >> (63 - i)) & 0x1;
        }
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 16; j++) {
            reg16[i] = reg16[i] << 1;
            reg16[i] ^= (qwIvReg >> (63 - i)) & 0x1;
        }
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 32; j++) {
            reg32[i] = reg32[i] << 1;
            reg32[i] ^= (qwIvReg >> (63 - i)) & 0x1;
        }       
    for (int t = 0; t < 10; t++)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) {
                bitsliced_keystream[t*8+i] = bitsliced_keystream[t*8+i] << 1;
                bitsliced_keystream[t*8+i] ^= (lpKsOut[t] >> (7 - i)) & 0x1;
            }
    for (int t = 0; t < 10; t++)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 16; j++) {
                bitsliced_keystream16[t * 8 + i] = bitsliced_keystream16[t * 8 + i] << 1;
                bitsliced_keystream16[t * 8 + i] ^= (lpKsOut[t] >> (7 - i)) & 0x1;
            }
    for (int t = 0; t < 10; t++)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 32; j++) {
                bitsliced_keystream32[t * 8 + i] = bitsliced_keystream32[t * 8 + i] << 1;
                bitsliced_keystream32[t * 8 + i] ^= (lpKsOut[t] >> (7 - i)) & 0x1;
            }
    cudaMalloc((void**)&gpu_awTea3LutA, 8 * sizeof(uint16_t));
    cudaMalloc((void**)&gpu_awTea3LutB, 8 * sizeof(uint16_t));
    cudaMalloc((void**)&gpu_abTea3Sbox, 256 * sizeof(uint8_t));
    cudaMalloc((void**)&gpu_lpKsOut, 10 * sizeof(uint8_t));
    cudaMalloc((void**)&gpu_captured_key, 2 * sizeof(uint64_t));
    cudaMalloc((void**)&bitsliced_keystream_d, 80 * sizeof(uint8_t));
    cudaMalloc((void**)&bitsliced_keystream16_d, 80 * sizeof(uint16_t));
    cudaMalloc((void**)&bitsliced_keystream32_d, 80 * sizeof(uint32_t));
    cudaMalloc((void**)&reg_d, 64 * sizeof(uint8_t));
    cudaMalloc((void**)&reg16_d, 64 * sizeof(uint16_t));
    cudaMalloc((void**)&reg32_d, 64 * sizeof(uint32_t));

    cudaMemcpy(gpu_awTea3LutA, g_awTea3LutA, 8 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_awTea3LutB, g_awTea3LutB, 8 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_abTea3Sbox, g_abTea3Sbox, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_lpKsOut, lpKsOut, 10 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_captured_key, captured_key, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitsliced_keystream_d, bitsliced_keystream, 80 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitsliced_keystream16_d, bitsliced_keystream16, 80 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitsliced_keystream32_d, bitsliced_keystream32, 80 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(reg_d, reg, 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(reg16_d, reg16, 64 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(reg32_d, reg32, 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    float time = 0;
    cudaEvent_t startx, stopx;
    cudaEventCreate(&startx);    cudaEventCreate(&stopx);    cudaEventRecord(startx);
    if (choice == 1) tea3_exhaustive << <BLOCKS, THREADS >> > (dwFrameNumbers, dwNumKsBytes, gpu_lpKsOut, gpu_awTea3LutA, gpu_awTea3LutB, gpu_abTea3Sbox, trials, gpu_captured_key);
    else if (choice == 2) tea3_exhaustive_0conflict << <BLOCKS, THREADS >> > (dwFrameNumbers, dwNumKsBytes, gpu_lpKsOut, gpu_awTea3LutA, gpu_awTea3LutB, gpu_abTea3Sbox, trials, gpu_captured_key);
    else if (choice == 3) tea3_exhaustive_1conflict << <BLOCKS, THREADS >> > (dwFrameNumbers, dwNumKsBytes, gpu_lpKsOut, gpu_awTea3LutA, gpu_awTea3LutB, gpu_abTea3Sbox, trials, gpu_captured_key);
    else if (choice == 11) tea3_exhaustive_bitsliced << <BLOCKS, THREADS >> > (reg_d, dwNumKsBytes, bitsliced_keystream_d, gpu_abTea3Sbox, trials, gpu_captured_key);
    else if (choice == 12) tea3_exhaustive_bitsliced_shared << <BLOCKS, THREADS >> > (reg_d, dwNumKsBytes, bitsliced_keystream_d, gpu_abTea3Sbox, trials, gpu_captured_key);
    else if (choice == 13) tea3_exhaustive_bitsliced16 << <BLOCKS, THREADS >> > (reg16_d, dwNumKsBytes, bitsliced_keystream16_d, gpu_abTea3Sbox, trials, gpu_captured_key);
    else if (choice == 14) tea3_exhaustive_bitsliced32 << <BLOCKS, THREADS >> > (reg32_d, dwNumKsBytes, bitsliced_keystream32_d, gpu_abTea3Sbox, trials, gpu_captured_key); // Best
    cudaMemcpy(captured_key, gpu_captured_key, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopx);    cudaEventSynchronize(stopx);    cudaEventElapsedTime(&time, startx, stopx);
    printf("Captured key: %llx %llx\n", captured_key[0], captured_key[1]);
    printf("Elapsed time: %f\n", time);
    printf("%s\n", cudaGetErrorString(cudaGetLastError())); 
    return 0;
}