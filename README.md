# CUDA TEA3

These CUDA Optimizations are used in the ToSC publication **GPU Assisted Brute Force Cryptanalysis of GPRS, GSM, RFID, and TETRA - Brute Force Cryptanalysis of KASUMI, SPECK, and TEA3**.

It measures how many seconds it takes for your GPU to perform **2^{18 + n}** key trials where **n** is a user input. 

In our optimizations we combined both the straightforward implementation technique and the bitsliced implementation technique. We obtained the best results when we used 32-bit registers and achieved 2^{34.71} key trials per second on an RTX 4090. This is around
160 times faster than our straightforward implementation.

Our best optimizations show that 80-bit key search for TEA3 would require 1.36 million RTX 4090 GPUs to break it in a year.

