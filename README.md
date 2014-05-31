efit2d-pyopencl
===============

Optimized OpenCL implementation of the Elastodynamic Finite Integration Technique for viscoelastic media


M. Molero-Armenta, Ursula Iturraran-Viveros, S. Aparicio, M.G. Hernández

Development of parallel codes that are both scalable and portable for different processor architectures is a challenging task. To overcome this limitation we investigate the acceleration of the Elastodynamic Finite Integration Technique (EFIT) to model 2-D wave propagation in viscoelastic media by using modern parallel computing devices (PCDs), such as multi-core CPUs (central processing units) and GPUs (graphics processing units). For that purpose we choose the industry open standard Open Computing Language (OpenCL) and an open-source toolkit called PyOpenCL. The implementation is platform independent and can be used on AMD or NVIDIA GPUs as well as classical multi-core CPUs. The code is based on the Kelvin-Voigt mechanical model which has the gain of not requiring additional field variables. OpenCL performance can be in principle, improved once one can eliminate global memory access latency by using local memory. Our main contribution is the implementation of local memory and an analysis of performance of the local versus the global memory using eight different computing devices (including Kepler, one of the fastest and most efficient high performance computing technology) with various operating systems. The full implementation of the code is included

