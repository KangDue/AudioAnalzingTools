# Project Specification: Ultra-Fast Audio Beam Focusing Simulator using Python

## Overview
Develop a Python-based audio beam focusing simulator for spatial sound analysis using a circular microphone array. The system targets real-time or close-to-real-time performance, using optimized fractional delay filtering and FFT-based convolution for efficiency. The simulator will generate energy maps representing focused audio on a target plane, enabling dynamic visualization of beam focusing changes over a 10-second time period.

## System Requirements

### Input Data
- **Audio Data**: Simulated or real, multi-channel (32 channels), duration of 10 seconds, sampled at 51,200Hz. Each channel represents recordings as if from a circular array microphones (radius=0.2m).
- **Geometry**:
  - **Microphone Array**: 32 microphones, arranged in a circular layout (radius=0.2m).
  - **Target Plane**: Square, 0.4m × 0.4m, located 0.4m away from array's center, discretized into a 70×70 grid (resolution).

### Processing Pipeline

1. **Fractional Delay Computation**  
   Calculate sub-sample delays from each grid point (on the target plane) to each microphone, using accurate geometric modeling. Utilize efficient fractional delay filters based on polyphase or Lagrange interpolation, specifically optimized for frequency-domain operation[1][2].

2. **FFT-Based Convolution**
   Convolve microphone signals and fractional delay filters in the frequency domain to maximize computational speed and leverage vectorized/numpy operations.

3. **Beamforming Algorithm**
   - Default to Delay-and-Sum Beamforming, but allow for the implementation of alternative fast algorithms (e.g., Minimum Variance Distortionless Response (MVDR), Fast Iterative Shrinkage algorithms, or spatial FFT-based techniques if benchmarks justify improved speed and accuracy[3][4][5][6]).
   - Incorporate per-grid-point or block-wise batching to exploit parallel computing (via NumPy or CuPy/GPU if available).

4. **Energy Map Visualization**
   - For each simulation time step (e.g., every 100ms or adjustable granularity):
     - Compute focused energy for all target plane grid cells.
     - Produce a 2D energy map overlay representing beam focus results.
   - Overlay beam energy maps onto the target plane image and animate/plot map evolution through time for source localization tracking.

5. **Performance Considerations**
   - All main computations (fractional delay, FFT convolution, beamforming) must be vectorized for batch processing.
   - Exploit memory and I/O optimizations; when possible, operate in-place or chunk large inputs.
   - Allow for optional integration with GPU via CuPy, Numba, or Torch for extreme speedup.

## Deliverables

- **Python package/source code** structured into logical, reusable modules, with clear documentation.
- **Configurable Simulator**: Main script or Jupyter notebook interface enabling input parameter tweaking (duration, sampling, grid size, beamforming algorithm choice).
- **Visualization**: Animated or interactive plots showing the energy map overlay on the target plane as it evolves over time.
- **Efficiency Benchmark**: Document/plots comparing baseline and optimized code, with suggestions for further speed improvements.
- **User/Dev Guide**: Instructions for running simulations, extending algorithms, and interpreting results.

## Reference Technologies & Practices

- Fractional delay filters and frequency-domain convolution for high-resolution beamforming[1][2].
- Modern Python simulation toolkits for array acoustics (see GSound-SIR for potential utility and parallelization example)[7].
- Efficient, open-source beamforming implementations for custom adaptation[6].

## Optional Enhancements

- Real-time GUI for live visualization.
- Integration of deep learning-based beamforming as a selectable mode.
- Export of simulation videos or data for further analysis.

For the development, focus on:
- **Maximal computational efficiency** (batch processing, FFT acceleration),
- **Accurate spatial energy representation**,
- **Clarity and extensibility for further research**.

References to core algorithmic efficiency and best practices: [1][2][7][4][5][6]

[1] https://www.osti.gov/servlets/purl/642743
[2] https://dl.acm.org/doi/pdf/10.1109/TASLP.2016.2631338
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC10098916/
[4] https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Tashev_MABeamforming_ICASSP_05.pdf
[5] https://www.spsc.tugraz.at/sites/default/files/Clenet10_DA.pdf
[6] https://github.com/schipp/fast_beamforming
[7] https://arxiv.org/html/2503.17866v1
[8] https://www.diva-portal.org/smash/get/diva2:833404/FULLTEXT01.pdf
[9] https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b9ef73cdf051f2cf89c63509d28ff749ff4f63a8
[10] https://new.eurasip.org/Proceedings/Eusipco/Eusipco2010/Contents/papers/1569292213.pdf
[11] https://israelcohen.com/wp-content/uploads/2020/03/09037110.pdf
[12] https://pysdr.org/content/doa.html
[13] https://kr.mathworks.com/matlabcentral/answers/559958-filtering-using-fft-for-audio-signal
[14] https://supersonic.eng.uci.edu/download/AIAA-2022-1154.pdf
[15] https://www.sciencedirect.com/science/article/abs/pii/S0022460X19306273
[16] https://github.com/KoljaB/RealtimeSTT
[17] https://pubs.aip.org/asa/jasa/article/156/1/405/3303425/A-circular-microphone-array-with-virtual
[18] https://www.sciencedirect.com/science/article/abs/pii/S0041624X23002573
[19] https://eprints.soton.ac.uk/421465/1/00392722.pdf
[20] https://www.mdpi.com/1424-8220/24/20/6644