# NOTE #
While Particle Filter library has been developed by me, the simulation code doesn't belong to me. I have just integrated my PF library into it for testing purposes. The simulation code has been developed by Steven Dumble. I have seen this simulation while following his beautiful Advanced Kalman Filter course in Udemy. Here is the link to the original repository: https://github.com/StevenDumble/AKFSF-Simulation-CPP

# Short Description of Project #
This is an attempt to create header only Particle Filter library. This code will be updated continually in order to make it more sophisticated and user friendly. Additionally, real-word simulation example has been added also in rder to clarify how this library works.

![simulation-gif](/simulation_gif.gif)

# More Details #
Library is header only and you can investigate it by looking at **folder _particle_filter_** inside **_include_ directory**. In order to investigate how this libarry should be utilized, please investigate **file _pf_functions.h_** which shows the kernel functions for initialization of the PF and predict and update steps based on the sensor type (gyro, gps, lidar in this case) and **_simulation.cpp_** which shows how to call PF functions correctly based on the sensor.
