---
title:	CS 267 Final Project --- Application of Parallelization in POLARBEAR's Pipeline for CMB Analysis through Cython
subtitle:	Spring 2017, University of California, Berkeley
author:
	- Kolen Cheung
	- Katherine Oosterbaan
date:	\today
abstract:	|
	We proposed a strategy to speedup the pipeline of POLARBEAR, a group that measures Comic Microwave Background Radiation (CMB) polarization, using Cython such that SIMD vectorization and OpenMP parallelization can be utilized relatively easily. 3 example functions are focused in this study that involves various different properties and hence different kinds of optimization techniques are required and has different benchmark behaviors.
	In the end, we found that the result is generally positive, with a projected speedup of $\sim 6$ times on Cori's Haswell nodes and $\sim 8$ times on Cori's KNL nodes. However, more investigation is needed to be certain, because of the variance in strong scaling behavior in different cases.
keywords:	CS267, Parallel Computing, POLARBEAR, CMB, final project
lang:	en
papersize:	letter
fontsize:	10pt
documentclass:	memoir
classoption:	oneside, article
geometry:	inner=1in, outer=1in, top=1in, bottom=1.25in
fontfamily:	fourier
usepackage:	siunitx,cancel,physics,placeins
hypersetup:	bookmarksnumbered,pdfpagelabels,pagebackref,hypertexnames=false,linktocpage=true
colorlinks:	true
linkcolor:	blue
citecolor:	blue
urlcolor:	blue
toccolor:	blue
header-includes: |
	\usepackage{siunitx}
...

# Introduction

## The Physics

Physicists always push the boundary of our understanding of the most fundamental aspects of the Universe. Some of the fundamental questions we can ask are what constitute the Universe; how gravity plays a role in quantum Physics; why neutrino mass are non-zero, how much mass do they have, and how many numbers of them; and if the current understanding of the Universe through $\Lambda$-CDM model is correct.

Many of such questions can only be answered when we probed at higher and higher energy scales. For example, the highest energy scale we can achieve artificially in the state-of-the-art LHC is about $\sim \SI{10}{\TeV}$, or $\SI{10e13}{\eV}$. But we can do only so much experimentally because of the limit of the size of the equipment we can build, and it is unlikely for the foreseeable future to create energy scale as high as the GUT scale at $\sim \SI{10e16}{\GeV}$, or $\SI{10e25}{\eV}$, which will be important for Quantum Gravity.

So instead of relying on human-built machine, one can measure the primordial signals created by the Universe itself. And the oldest possible such signal that is observable is the Cosmic Microwave Background (CMB) Radiation. It is the first light of the Universe when it was $\sim 400000$ years young, and everything happened between now and then are imprinted in this signal. Some of the information we can extracts includes gravitational wave at GUT scale (by B-mode analysis on the CMB), dark matter, neutrino mass, falsification of $\Lambda$-CDM model, etc.

POLARBEAR is one of the pioneer group on the measurement of CMB polarization in University of California, Berkeley. One of the major result by POLARBEAR in 2014 is[@Collaboration:2014eg]:

> the hypothesis of no B-mode polarization power from gravitational lensing is rejected at $97.2\%$ confidence

![B-mode](media/CMB-B-mode.pdf){width=80%}\ 

## The Pipeline

In order to measure the CMB, a telescope is scanning the sky with $\sim 100$ sensors, each taking a time stream data. An output is created per hour-long observation, called Constant Elevation Scan (CES). There are about $\sim 14$ such observation per day, and after about 2 years of observation, we accumulate about $\sim 10,000$ CES.

These data requires a lot of cleanup, partially due to the imperfection of the equipments, and partially due to the uncontrollable factors like weather and atmospheric conditions. In the end, $4237$ such inputs are used for the analysis.

Since the correlation time scale is $\sim \SI{20}{\min}$, shorter than the observation time scale, and we can treat each observation independently. Hence, it is a perfectly (embarrassingly) parallel problem.

In the current implementation of AnalysisBackend, the code is mainly written in Python with some occasional C/C++ modules, and relies heavily on external libraries e.g. Numpy. All code is written in serial, and in the end a trivial MPI process is run to start these individual independent processes.^[MPI is used to avoid the Python start-up overhead only. i.e. There is no communication between processes at all.]

However, such a perfect parallel treatment is not without its problems:

1. Cori's Haswell nodes has only $128$ GB RAM, this limits the number of concurrent in-node processes to about $16$, which is significantly less than the $32$ number of cores available.

2. Due to the MASTER algorithm [@Hivon:2001eh], in the past CMB analysis is mainly $O(n^3)$ which enjoys high computational intensity. The MASTER algorithm improve the scaling to $O(n)$, but this decreases the computational intensity and therefore the pipeline is highly likely to be IO bounded. It also means that it potentially will be speeded up by hyper-threading.

3. Ideally, we would want to run the pipeline using Cori's Knights Landing (KNL) nodes. This means that it is going to be slightly more RAM-limited, and have much more cores and hyper-threading available. Currently, KNL can afford at most $272$ threads to run concurrently, much bigger than the current $16$ processes per node we are using.

4. SIMD vectorization is also important, and is currently unexplored in AnalysisBackend. It is because Haswell CPU has 4-wide and KNL has 8-wide Fused multiply–add (FMA).

It is worth noting that the Intel Distribution for Python which is also used on NERSC has been optimized for parallelization as well:

> The Intel Distribution accelerates performance of Python packages with Intel® Performance Libraries, including Intel® Math Kernel Library (Intel® MKL), Intel® Threading Building Blocks (Intel® TBB), Intel® Data Analytics Acceleration Library (Intel® DAAL), and Intel® MPI. The packages have been optimized to take advantage of parallelism through the use of threading, multiple nodes, and vectorization. The release notes includes a full list of the packages included in the distribution. From [Anaconda & Intel Python Distribution FAQ](https://www.continuum.io/sites/default/files/AnacondaIntelFAQFINAL.pdf).

> The all-included, out-of-the box distribution accelerates core Python packages including NumPy, SciPy, pandas, scikit-learn, Jupyter, matplotlib, and mpi4py. It integrates the powerful Intel® Math Kernel Library (Intel® MKL), Intel® Data Analytics Acceleration Library (Intel® DAAL) and pyDAAL, Intel® MPI Library, and Intel® Threading Building Blocks (Intel® TBB). From [Intel® Distribution for Python* | Overview | Intel® Software](https://software.intel.com/en-us/intel-distribution-for-python#close).

Since most of the existing code heavily relies on Numpy, and from the description above one would expect free parallelization from Intel's Distribution for Python. This turn out not to be the case at least in our study. See more in the [benchmark of Polynomial Filter Array][Polynomial Filter Array] in the Appendix. More investigation is needed to see what kind of parallelization can be expected from the libraries in Intel's Distribution for Python.

# Optimization and Parallelization

From the above limitation, we laid out the following requirements:

1. SIMD vectorization and OpenMP parallelization is necessary, for both Cori's Haswell and KNL nodes, and more so on the later one.

2. (non-trivial) MPI^[trivial MPI refers to the use of MPI mentioned above that has no communication at all.] is not necessary however, because each CES is still independent, and in the foreseeable future the RAM needed for a single CES will not surpass the RAM available in 1 node on Cori.

3. Written in Python. This has been mostly true in AnalysisBackend. The choice of Python can be attributed to historical reasons, but also because of maintainability (e.g. in writing, packaging, distributing, testing) and readability.

## The Proposal

Our proposal is then to write every module in Cython.

Cython as a language is a superset of Python, and as a compiler is a transpiler to C/C++. It supports both SIMD vectorization and OpenMP parallelism. Many but not all C/C++ features are supported.

Cython is not without its limitation. Directive pragma and Intel intrinsics are not supported, SIMD vectorization relies on the C/C++ compilers' optimization, and obtaining the vectorization report is convoluted^[typically Cython-tranpiled C++ code is on the order of $100$ times longer than the original Cython code.]. OpenMP support is limited, e.g. `#pragma omp for` can be used in Cython by `prange`, reduction is implied by in-place operators, and as a result other kinds of reduction like `max` are not supported.

Fortunately, these limitations will not be important to POLARBEAR's analysis, as shown in the demonstration below. One of the reason is that there are a lot of inherit parallelism in the pipeline. For example, each channel from the data are usually independently processed, which is a trivial use case of `prange`.

# Results, Benchmarks, and Discussions

As a proposal on parallelizing the whole pipeline, we focuses on 3 functions in the pipeline to demonstrate different aspects of the difficulties involved and the potential speed gain from it. They are (more details will be given below):

1. Ground template filter: was originally written in Python with Numpy and C with Weave^[Weave, formerly Scipy.weave, is a now deprecated tool to allow easy writing of C-modules in Python. Note that the Scipy team has migrated their own Weave codebase using Cython.]. This is a testbed to see how to fully Cythonize a Python function, i.e. written in C-style with Cython syntax without the use of Numpy at all.

2. Polynomial filter: was originally written in pure Python with Numpy. This is a testbed on the potential speed gain on a function that is highly pythonic and heavily relies on Numpy that full Cythonization is virtually impossible (without reinventing the wheels from Numpy or using similar libraries in C/C++).

3. Boundary distance function: was originally written in Python with Numpy and C with weave. This is one of the major bottleneck of the whole pipeline, and is the testbed to see how much speed up we could get from Cythonization in an actual application hotspot.

Ground template filter and boundary distance function are both rewritten completely Cythonic^[Cythonic is defined as oppose to Pythonic: Cythonic code is like writing C/C++ using Python syntax.] without the using of Numpy functions^[except when creating and returning Numpy array as a Python object]. For the polynomial filter, it is minimally Cythonized because of the heavy reliance on Numpy's functions. Since there is virtually no speed gain from the OpenMP parallelization in the polynomial filter, it will be mentioned in the appendix. In the following 2 sections, we will focus on the ground template filter and boundary distance function. All the codes can be found in [GitHub - ickc/TAIL](https://github.com/ickc/TAIL).

The following benchmarks are done on Cori's Haswell nodes. Although we want to use the KNL nodes to push the boundary of vectorization and OpenMP scaling, Cori's KNL queue has been impenetrable on the days near the submission of this report. In the near future we will profile it on KNL.

## Ground Template Filter

The actual code for this filter can be found in [ickc/TAIL/ground_template_filter_array.pyx · GitHub](https://github.com/ickc/TAIL/blob/master/tail/timestream/ground_template_filter_array.pyx).

### Description

First, we scan the sky in constant elevation with a sweeping azimuth:

![$\theta (t)$](media/az.pdf){width=55%}\ 

And there is a boolean mask that select the data and discard poor data:

![Mask$(t)$](media/mask.pdf){width=55%}\ 

And finally the signal:

![$d(t)$](media/signal.pdf){width=55%}\ 

The mask is for data selection. So the main focus here are the signal $d(t)$ and the azimuth $\theta(t)$. These 2 equations can be considered as a set of parametric equations in time $t$. And the ground template filter is basically to find the $\langle d(\theta) \rangle$, the average signal as a function of $\theta$.

In the algorithm, this is done through a given resolution $n$ (say, $300$ pixels), the range in $\theta$ is divided into $n$ bins, and the signal is averaged into these bins.

### Optimization Procedures

#### Vectorization

The first step is to look at the HTML report which can be found in [Cython: ground_template_filter_array.pyx](https://ickc.github.io/TAIL/tail/timestream/ground_template_filter_array.html), and below we have an excerpt of it:

![Cythonization report](media/cythonization-report.pdf){width=60%}\ 

The key to look for in these reports are the "yellow-ness". More yellow means more Python interactions, and if it is white as in above, it is transpiled into pure C/C++ and Python's Global Interpreter Lock (GIL) can be released, which is important for OpenMP parallelization.

The next step is to check for vectorization and see if there is anyway to rewrite it to trigger vectorization. For example, from the same code above, if you click at the `+` sign in line 65 in the HTML version, you can see the C/C++ code it is translated into.

From [`ickc/TAIL/tail/timestream/ground_template_filter_array.cpp`](https://ickc.github.io/TAIL/tail/timestream/ground_template_filter_array.cpp), we can then find where this piece of C/C++ code is in the generated `.cpp` file. Trace it back to the nearest for-loop, which is in line 2303.

Finally, you can go into the [Intel vectorization report `ground_template_filter_array.optrpt`](https://ickc.github.io/TAIL/build/temp.macosx-10.6-x86_64-3.5/tail/timestream/ground_template_filter_array.optrpt) and find the reference of line 2303, and it reads

>     remark #15344: loop was not vectorized: vector dependence prevents vectorization

In this case, we will replace the code from line 63-66 by 69-71. Repeating the whole exercise and the report will say

>     loop was not vectorized: vectorization possible but seems inefficient.
>     Use vector always directive or -vec-threshold0 to override 

In this instance, rewriting the code to remove the branching statement made vectorization possible, but still inefficient and the compiler didn't vectorize it. It is worth noting that the code in line 69-71 is slower than the one in line 63-66. Without checking the report, one would write as in line 69-71 to hope for vectorization but actually get a slower code.

In other instances of the code, such kind of optimization paid off. One such case is to use the Conditional (ternary) Operator to rewrite an if-statement.

Another interesting instance in this particular problem is

```python
for i in range(size):
	bins_hit[i] = EPSILON
...
## SIMD checked
for k in range(nPix):
	# won't be 0 since it is initialized as EPSILON
	# if bins_hit[nBin * i + k] != 0:
	bins_signal[nBin * i + k] /= bins_hit[nBin * i + k]
```

In order to remove the special case of dividing by $0$, `bins_hit` is declared to be `double` instead of `int`, and initialized to be the machine epsilon rather than $0$. In the worst case scenario it makes an error of the order $10^{-16}$, but is going to speed up this part of the code a lot through SIMD vectorization. Since we know that in our application such a small error is acceptable, this will be a good trade off for us.

Of all the vectorization effort, one feature we missed is the ability to control the alignment of the memory. Numpy array are created in $16$-bytes alignment, which will be provided as the input of the function. But in order to fully optimized for vectorization, $64$-bytes alignment is needed. Even worst, even if it is $64$-bytes aligned^[which can be achieved by copying the data, `malloc`, and some pointer arithmetic, all available in Cython (but `aligned_alloc` is not).]), compiler hints through directive pragma is not support in Cython and therefore no speed up can be done by alignment.

#### Locality

In this example, it is also worth mentioning locality is also important. It is because the computation is of low intensity, the problem is IO bound and we want to minimize the time spent in IO.

From the description above, we see that the $\theta (t)$ is sweeping back and forth, and the signal $d(t)$ is averaged per bin in $\theta$. i.e. as it is swept through, either you have the locality of $d(t)$, or the locality of the bins of $\theta$, but not both.

Furthermore, this function has a key `lr`, when true, the averaged signaled is calculated separately when $\theta$ is sweeping left or right. In the original version of the code, this is done in 2 passes, where each pass has a mask to remove the signal that move in another direction (e.g. for left pass, right moving $\theta$ are removed by a mask). This is going to be RAM inefficient and slow because the IO on the signal is twice as long and require a mask as large as the original signal.

When such conflicts exists, the locality is prioritized towards the signal rather than the bins. Because the total number of bins is much less than that of the signals, so the bins has a higher chance to stay in a higher level of cache (say L3), whereas the signal is typically larger than L3 cache. One such example is this code:

```python
for i in prange(nCh, nogil=True, schedule='guided', num_threads=num_threads):
    # calculate ground template
    ## add total signal and no. of hits
    for j in range(nTime):
        if mask[i, j]:
            k = pointing[j]
            if isMovingRight[j]:
                bins_signal_r[nBin * i + k] += input_array[i, j]
                bins_hit_r[nBin * i + k] += 1
            else:
                bins_signal_l[nBin * i + k] += input_array[i, j]
                bins_hit_l[nBin * i + k] += 1
```

In the if-statement above, priority of locality is given to the signal `input_array`. If we wanted to favor the locality of the bins, the 2 branches should be split into their own for-loop.

#### OpenMP

The same example above also shows the use of OpenMP. The `nogil` is important here. Only region with no "yellow-ness" in the Cython to C/C++ conversion can releases the GIL. So if one wants to use OpenMP parallelization, then the content within the loop needs to be free of Python interaction.

This example also shows that the `i`-loop that runs through the `nCh`, stands for the number of channels, are independent. This is often true in the filters and will be trivial to parallelized.

### Benchmark

With number of channels $100$, number of time-stream signal $10000$, and number of bins $300$, `lr=True`, the ground template filter in the old pipeline takes $\SI{7.53}{\ms}$ to complete, and our code takes $\SI{1.96}{\ms}$ to complete. i.e. a speed up of $3.84$ times, just from Cythonization and SIMD *without* OpenMP parallelization yet.

As we add more numbers of concurrent threads, the scaling looks like the following, using number of channels $50,000$, number of time-stream signal $10,000$, and number of bins $8,192$, `lr=True`:

![Strong scaling coefficient number of channels $50,000$, number of time-stream signal $10,000$, and number of bins $8,192$](media/ground_template.pdf){width=85%}\ 

It can be seen that this does not scale very well beyond $8$ threads, and hyper-threading actually will make it slower. The reason is that even for this unrealistically large no. of channels, the computation intensity is still very low and the OpenMP overhead is relatively high when $p$ is large --- it is likely that all threads are accessing the L3 cache where the bins mostly lived and the RAM where the signal mostly lived and are congressed. i.e. in the roofline model we are on the way left side of the roof.

Fortunately, since we are starting $16$ processes. When Haswell/KNL nodes are used, the number of threads is $2$/$4$ respectively (if hyperthreading is not used as it is shown to be not helping). In this region,

![Strong scaling coefficient number of channels $50,000$, number of time-stream signal $10,000$, and number of bins $8,192$](media/ground_template-2.pdf){width=85%}\ 

we see that the strong scaling coefficient is $0.406$, meaning that the time needed scales as $\frac{1}{p^{0.406}}$. As mentioned above, the low coefficient is explained by the low computational intensity.

## Boundary Distance Function

The actual code for this filter can be found in [ickc/TAIL/boundary_distance.pyx · GitHub](https://github.com/ickc/TAIL/blob/master/tail/timestream/boundary_distance.pyx).

### Description

This function is much easier to be described, involving well-known algorithm in Computer Science: boundary tracing algorithm and post-office problem.

The function takes a $n\times n$ 2D array, which has 1 simply-connected region of `True` value, and elsewhere `False`. It can be visualized as 1 single island of `True` value around a sea of `False` in a square map^[Physically, the `True` island represent the map we have for the CMB. Since we are scanning for a curvilinear sky, the map we made is not necessarily rectangular.].

What this function does is for every point in the island, find the distance to the closest point on the boundary. A 2D array of these distances are returned as a Numpy array.

We implemented it by first using a Boundary tracing algorithm to detect the boundary first. After that, it is a post-office problem — for every point inside the boundary, determine the closest point on the boundary. Hence this is an $O(n^3)$ problem.

### Optimization Procedures

The vectorization and OpenMP parallelization procedure is the same as that laid out in [Ground Template Filter] and we are not repeating it here. The Cythonization report is in [Cython: boundary_distance.pyx](https://ickc.github.io/TAIL/tail/timestream/boundary_distance.html) (and you can see most Python interaction is avoided), and the generated `.cpp` code is in [ickc/TAIL/tail/timestream/boundary_distance.cpp](https://ickc.github.io/TAIL/tail/timestream/boundary_distance.cpp), and the [vectorization report is in here](https://ickc.github.io/TAIL/build/temp.macosx-10.6-x86_64-3.5/tail/timestream/boundary_distance.optrpt).

One example that shows both SIMD vectorization and OpenMP parallelization is

```python
for i in prange(x_min + 1, x_max, nogil=True, schedule='guided', num_threads=num_threads):
    for j in range(y_min + 1, y_max):
        if mask[i, j]:
            loc = i * m + j
            # SIMD checked
            for k in range(nBoundary):
                distance_sq = (i - boundary_coordinate[2 * k])**2 + \
                    (j - boundary_coordinate[2 * k + 1])**2
                distances_sq[loc] = distance_sq \
                    if distance_sq < distances_sq[loc] else distances_sq[loc]
```

This is the post office problem, where it is determining the distances of each points to the boundary, and is the part that the function is going to spend the majority of time in because it is $O(n^3)$. In the innermost `k`-loop, it is vectorized using the ternary conditional operator, without which the if-statement would have prevented vectorization. In the outermost `i`-loop, OpenMP-for is used, where `i` represent the `i`-th row of the pixels on the map, which are independent of each other.

Another (counter) example is

```python
# convert boundary from 1D indexing to 2D indexing
# and obtain the smallest box that includes the boundary
# Not vectorized, use the following if in C
#pragma ivdep
for k in range(nBoundary):
    x = boundary[k] // m
    y = boundary[k] % m
    boundary_coordinate[2 * k] = x
    boundary_coordinate[2 * k + 1] = y
    x_min = x if x < x_min else x_min
    y_min = y if y < y_min else y_min
    x_max = x if x > x_max else x_max
    y_max = y if y > y_max else y_max
```

Because of the apparent vector-dependance, it is not vectorized. If it were written in C/C++, we would have used a `#pragma ivdep` at this point. Fortunately it is $O(n)$ and is not the hotspot of the application.

It is worth mentioning that the boundary tracing algorithm is not parallelized nor anywhere has SIMD vectorization in it. This part of the code is essentially having a class of object `Turtle` that walks down the map and keep tracing the boundary until it goes back to the starting point.

Vectorization is impossible in the boundary tracing algorithm, but OpenMP parallelization is at least in theory possible. We could have divided the map into $p$ regions and starts "workers" to trace the boundaries in parallel and later merge them together. However, it will be challenging to merge the individual boundaries together to make it becomes one big boundary, while remains robust.

The reason we choose not to parallelize this region is because this algorithm is $O(n)$ (where the input mask is of size $n\times n$), and since it is not optimizable, potential speed-up from even KNL is at most $68$ times (and in reality very much less because we are starting $16$ processes at once), without even account for the time to combine all those boundaries.

However, we need to stress that this possibility is not ruled out. If in the future this becomes a hotspot, such parallelization scheme should be investigated. In fact, although we wrote this part of the code using object-oriented programming, we used Cython's Extension types such that they are not really Python classes but are translated into structs and functions. i.e. All Python interaction is removed and GIL can be released, which is the cornerstone of using OpenMP in Cython. A bonus of this design is that there is no Python overhead and the algorithm is very fast.

### Benchmark

We created a mask of $1024 \times 1024$ large, and an island of radius $128$ in the middle. We run the benchmark on the original AnalysisBackend algorithm at $\SI{583}{ms}$, and our algorithm at $\SI{119}{ms}$, i.e. a speed up of $4.90$ times, just from Cythonization and SIMD *without* OpenMP parallelization yet.

In the strong scaling test, we changed the radius to $510$, and the result is

![Strong scaling of boundary distance function](media/boundary_distance.pdf){width=85%}\ 

we see that it is almost perfectly parallelized with a strong scaling coefficient of $0.909$, meaning that the time needed scales as $\frac{1}{p^{0.909}}$. This is close to ideal ($1$), and is expected because the rows of pixels are totally independent and hence is perfectly parallel.

# Conclusion

Speedup is achieved through

1. Cythonization
2. SIMD vectorization
3. OpenMP parallelization

It is hard to decouple the 1st and 2nd effect. For the combined effect of Cythonization and SIMD vectorization, in the worst case scenario which involved minimal changes to the Python code hence remains Pythonic, the [Polynomial Filter Array] in the appendix, has $3.23$ speedup. In the best case scenario where Python interaction is minimized without using Numpy functions, the [Boundary Distance Function], a speedup of $4.90$ is achieved.

For the 3rd effect, [Polynomial Filter Array] which stays Pythonic and relies on Numpy has virtually $0$ efficiency. More investigation is needed here, as mentioned in the appendix. For the [Ground Template Filter], the strong scaling coefficient is $0.406$ because of low computational intensity, and fortunately is not the hotspot of the application^[in actual data size, it finishes on the order of $\si{\ms}$.] Lastly, in the [Boundary Distance Function], high strong scaling coefficient is achieved at $0.909$. This will be important since this is a hotspot for the application.

To give a very rough estimation on the potential speedup for the whole pipeline, we will give a factor of $4$ for Cythonization and SIMD vectorization. We guesstimate the strong scaling to be ~$0.5$, with a huge grain of salt that it varies a lot through these 3 different functions. From this preliminary study, hyper-threading does not help and so we are not considering this. Since we use $16$ process per node, for Cori's Haswell nodes, we would use $2$ threads per process, and for KNL, we would use $4$ threads per process. The overall estimated factor of speed up will then be:

- Haswell nodes: $\sim 6$
- KNL nodes: $\sim 8$

For a job that would have required $\sim 220,000$ NERSC hours on Haswell, it would becomes $\sim 40,000$. It should be emphasized that this is a very rough estimation though, and more detailed analysis is needed in the future.

\appendix

# More on Boundary Distance Function

The boundary distance function is actually part of a larger code called the pseudo-power spectrum. Boundary distance function is needed for an apodization mask, which essentially filter the map such that the "island" has a smooth transition to zero on the boundary.

This apodization mask might be created by an $O(n)$ algorithm instead (whereas the boundary distance function is $O(n^3)$), essentially by walking through the boundary $k$-times, where $k$ is the width of the apodization mask.

This has not been done in this study, because we find it very hard to parallelized a boundary tracing algorithm. One suggestion is made in [Boundary Distance Function] and might be investigated in the future.

However, this alternative proposal on the apodization mask might turns out to be more effective, even when run in serial, because as we are making higher and higher resolution map, $O(n)$ is always going to win the $O(n^3)$ algorithm.

More investigation is needed in this comparison. And in fact a quick demo based on the implementation of boundary distance function in this report is possible. Because a boundary tracing algorithm is already implemented in this study, and the proposed apodization mask is basically an iteration of the boundary tracing $k$ times.

# Polynomial Filter Array

## Description

The poly\_filter\_array code uses a Legendre polynomial to fix the gain of the data, as the gain drifts during the scanning period. Unlike ground\_template\_filter\_array , poly\_filter\_array relies heavily on Numpy functions, which for pure Python code provides good speedup. However, it is very unfortunate when one is trying to transition the code to Cython.

To transition the code to Cython, we rewrote the code in the Cython memoryview array style, similar to the ground template filter. However, when it came time to parallelize, we ran into serious problems. To parallelize Cython code, it is required that we release the GIL, which is not possible when code contains Numpy calls. Unfortunately, this was almost the entire code, which meant that the few cases where we actually could call prange resulted in no speedup whatsoever. For a future direction, we recommend rewriting the code to eliminate the use of Numpy, which would enable the code to be reformatted in C-style and parallelized, which is the maximum Cython speedup.

One more point of interest with this code is that it was written and compiled in a Jupyter notebook that was run within NERSC. NERSC uses the Intel MKL distribution of Python and Numpy, which actually already uses OpenMP to deliver thread-level parallelism and include specialized vectorization instructions. This means that simply by compiling and running the code on NERSC, we may already be taking advantage of parallelism that’s implicitly built into the Python framework.

In summary, when code contains significant amounts of Numpy operations, optimal conversion to Cython and Cythonized parallelization is not possible. To fully implement code in Cython and to parallelize it, code must be rewritten to eliminate Numpy calls.


## Benchmark

For number of channels $10,000$, number of time stream data point $10,000$, and polynomial order $4$:

In the original code, it took $\SI{1.14}{\s}$ and in the cythonized code, $\SI{353}{\ms}$ in serial. i.e. $3.23$ times improvements.

However, there is virtually no difference in timing when number of threads is increased. This is unexpected because as mentioned in [The Pipeline], parallelization should be built-in from the Numpy provided by Intel's Distribution for Python.

More investigation is needed here in the future, and perhaps requires a deeper understanding of what kind of parallelism is provided in Intel's Distribution of Python.

Because there is no parallelization scaling in this function (the coefficient is virtually $0$), we decided not to put this part in the main report but in the appendix.

# Packaging and Distributing

Our modules is built using `setup.py`. All the compiler args can be put in `setup.py`, including the `-O` flag levels, `-mtune`, `-march` flags, etc. More investigation is needed on how to target different compilers at the same time. For instance, Clang compiler does not recognize the `-fopenmp` flag.

Currently, Intel's `icpc` compiler and GNU's `g++` compiler is supported. We found that `g++` compiler does not scale very well for OpenMP when there are too much threads than the number of jobs in `prange`. And for our codes, generally the `icpc` compiler does better at optimizing for speed.

Packaging is going to be important for our pipeline. In fact one of the forte of Python is packaging and distributing of the code. In the future we should investigate on using Conda build recipes to improve packaging and distribution, and also docker and shifter for both ease of deployment and potential speedup^[NERSC's study has shown that Shifter can accelerate Python applications due to the huge metadata IO that Python needs. See more in [Shifter: User Defined Images](http://www.nersc.gov/research-and-development/user-defined-images/) and [An Introduction to Python at NERSC](https://www.nersc.gov/assets/Uploads/10-Python.pdf).]. One potential challenge here might be the use of `icpc` compiler.

We should also mentioned that docstrings and comments are emphasized in TAIL, which is lacking in AnalysisBackend. In the future, these docstrings can easily be turned into documentation through a very popular Python library called Sphinx. We strongly believed that good documentation should play an important role in distributing our code.

Continuous Integration is also employed using [Travis](https://travis-ci.org/ickc/TAIL). Python 2.7 and Python 3.6 are test against, and the Cythonized C++ code is built using GNU's `g++` compiler. Intel's `icpc` compiler for Continuous Integration is possible[^[nemequ/icc-travis: Script to help install Intel C/C++ Compiler on Travis CI.](https://github.com/nemequ/icc-travis)] but has been unreliable to the author and not used. However, every build is tested locally using Intel's compiler before pushing to guarantee the code is always in a working state.

# Intel TBB, MPI

In the situations that involve more complicated parallelization, MPI and Intel Threading Building Blocks (TBB) might be beneficial. We did not try either approach here, but Intel has some nice articles on using MPI and TBB in Python:

- [Unleash the Parallel Performance of Python* Programs | Intel® Software](https://software.intel.com/en-us/blogs/2016/04/04/unleash-parallel-performance-of-python-programs)
- [Exploring MPI for Python* on Intel® Xeon Phi™ Processor | Intel® Software](https://software.intel.com/en-us/articles/exploring-mpi-for-python-on-intel-xeon-phi-processor)  

In particular, Intel TBB can becomes important if different levels of parallelization is applied where over-subscription might arise. Currently, all 3 functions involved in this study has incorporate a key-value function argument of `numthreads` default to be 4. If function composition is used, these `numthreads` should be set properly to avoid oversubscription. One possible alternative will be setting `numthreads` with the environment variables `OMP_NUM_THREADS` and uses Intel TBB to get around with oversubscription whenever it arises.

# Numba

One of the options that we pursued for code speedup/parallelization was Numba. Numba is a module that generates optimized machine code from a python codebase using the LLVM compiler structure. LLVM is a single static assignment(SSA)-based compilation structure. It allows you to just-in-time compile Python code (including Numpy) at import time, runtime, or statically. To test the speedup, we ran a nontrivial test function; the Numba additions are in red (jit is just-in-time compiling).

```python
import numpy as np
from numba import jit

@jit
def naive_convolve(f, g):
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2*smid
    ymax = wmax + 2*tmid
    h = np.zeros([xmax, ymax], dtype=f.dtype)
    for x in range(xmax):
        for y in range(ymax):
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h
```

In normal python without Numba, this code takes 1.41s to run with a 100x100 and 8x8 array for f and g. With Numba and jit, this code takes 2.8 ms to execute, a speedup of more than 40x, but times depend on the machine used. It has been reported that Numba can speed code up up to 200x (https://en.wikipedia.org/wiki/Numba).

When we began looking into Numba, we were under the impression that there was an additional function called prange, which uses as many CPUs as detected by the multiprocessing module and runs loops in parallel. However, we then found out that this function had been removed and Numba no longer possesses parallelization options. At this point, we abandoned this strategy, but it is worth mentioning because of the incredible (non-parallel) speedup obtained for Python code with very minimal effort.

# Reference
