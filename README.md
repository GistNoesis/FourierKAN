# FourierKAN

Pytorch Layer for FourierKAN

It is a layer intended to be a substitution for Linear + non-linear activation

This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
It should be easier to optimize as fourier are more dense than spline (global vs local)
Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
Avoiding the issues of going out of grid

# Usage
put the file in the same directory 
then 

```from fftKAN import NaiveFourierKANLayer```

alternatively you can run 
```python fftKAN.py```

to see the demo.

Code runs, cpu and gpu, but is untested. 

This is a naive version that use memory proportional to the gridsize, where as a fused version doesn't require temporary memory

# Highlight of the core :
You can either do the simple thing of materializing the product and then do the sum, or you can use einsum to do the reduction.
Einsum should use less memory but be slower

https://github.com/GistNoesis/FourierKAN/blob/9a8c75311b74ef9a858020edcc29e1a2059cb41e/fftKAN.py#L28-L58

# License 

License is MIT, but future evolutions (including fused kernels ) will be proprietary. 

# Fused Operations 
This layer use a lot of memory, but by fusing operations we don't need any extra memory, and we can even use trigonometry tricks.

https://github.com/GistNoesis/FusedFourierKAN
