# Laser-Speckle-Contrast-Imaging
Internship project

In this project we worked on image processing algorithms for fluid flow gradient parameters extraction.
In the begining we learned about the physical phenomenon of laser speckle and based on its statistical characteristics we created
simmulations of static and evolving in time speckle patterns.
Simulated data was used for developing contrast imaging algorithms in spatial and temporal domains using different kernels.

Second half of the project was done at the lab, we built a setup for simulation of capillary blood flow.
Our imaging setup comprised of focusing optics, a variable diaphragm and a CMOS sensor (XIMEA).
The area of interest was illuminated with a coherent laser.
Different imaging techniques were implemented in real time, speckle contrast techniques demonstrated their efficiency as 
a non-invasive real time tool for measuring particle velocity in a phantom tube.

For a more in-depth description please see `Paper.pdf`

Functions:
`speck_gen.m` - generates speckle pattern with desired paramters

`SpeckleSize.m` - Calculates speckle size for real speckle pattern 

`evolving_speckle.m` - generates evolving speckle pattern with desired paramters: speckle size, level of blurring, decorrelation time. Implements
different imaging techniques on simulated data.

`real_time.py` - was used for real time imaging with implementations of different techniques
