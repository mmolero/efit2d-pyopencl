#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@mainpage

Manuscript Title: Optimized OpenCL implementation of the Elastodynamic Finite Integration Technique for viscoelastic media

@authors
M Molero, U Iturraran-Viveros, S Aparicio, M.G. Hernández 

Program title: EFIT2D-PyOpenCL

Journal reference: Comput. Phys. Commun.

Programming language: Python.

External routines:  numpy, scipy, matplotlib, glumpy, pyopencl

Computer: computers having GPU or Multicore CPU with OpenCL drivers.


@package main

Usage: python main.py

The main procedure to run a simulation is as follows:

	- Create a Scenario
	- Define Boundary Conditions
	- Define Materials
	- Define Input Signal, (Transducer properties)
	- Define type of inspection: e.g. Transmission, Pulse-Echo
	- Define Simulation Parameters
	- Run



"""

import	 sys
import	 time
import	 numpy	              as		 np
import	 matplotlib.pyplot	  as        plt
import	 glumpy

#import all EFIT2D objects
from	 EFIT2D	              import *


#Flag for enabling Plotting
Plotting	    = False

#Flag for enabling Video Generation from Snapshots, only works when plotting is True
EnableVideo     = False

#Flag for enabling the Viscoelastic Computation. If the flag is set to False the Elastic computation will be done                  
VISCO		    = True

#Define the device used to compute the simulations
#       "CPU"        : uses the global memory in th CPU
#       "GPU_Global" : uses the global memory in the GPU
#       "GPU_Local"  : uses the local memory in the GPU
DEVICE		    = "GPU_Global"	        # CPU, GPU_Global, GPU_Local

#Output file Definition, the output file will be saved in .mat format (Matlab)
if VISCO:
	FILE    = "test_visco_"   + DEVICE  # FILE name, 
else:
	FILE	= "test_elastic_" + DEVICE  # FILE name, 

#Local Size    
Local_Size	 = (16,16)
#Simulation Time (in sec.)
Time		 = 50e-6

#Frequency of the Source
frequency    = 500e3
#Spatial Scale Factor:
#     SpatialScale = 1    -> meters
#     SpatialScale = 1e-3 -> millimeters  
SpatialScale = 1e-3   



if __name__ == '__main__':


	global start #Global variable for timing the simulations


	#Create Main Scenario
	#    width, height in mmm if SpatialScale = 1e-3
	#    width, height in m if SpatialScale = 1
	#    Pixel_mm -> pixel per millimiter or meters
	image = NewImage(Width=100, Height=100,Pixel_mm=10,label=0)	
	#Create a middle layer # centerW, centerH, Width, Height in mm if SpatialScale = 1e-3 or if SpatialScale = 1 in meters
	image.createLayer(centerW = 50, centerH = 50, Width=100, Height=50, label=60)
    	

	#Setup boundary conditions
	#  Tap -> size of layers
	image.createABS(Tap = 5)
	# Configure if boundary layers will be treated as absorbing layers or air layers
	#   False: Absorbing layers
	#   True : Air boundaries
	image.AirBoundary = False  
	
	
	#Definition of Materials
	# The Materials are stored in a List, each material is defined as an instance of the Material Class.
	# It is necessary to define the label of each material in order to handle them.
	materials = list()
	
	#Medium 1
	name  = 'medium1';
	rho	  =  2000; VL	= 1800; VT = 1040
	lam		  = rho*( VL**2 - 2*(VT**2) ); mu= rho*( VT**2); 
	eta_v =  6.8358     # (Pa s) bulk viscosity
	eta_s =  13.7672    # (Pa s) shear viscosity
	material1 = Material(name=name,rho=rho,c11 = lam + 2*mu,c12=lam,c22=lam + 2*mu,c44=mu,eta_v= eta_v, eta_s=eta_s,label=0)
	#Append to the material list
	materials.append(material1)

	#Medium 2
	name  = 'medium2';
	rho	  =  2600; VL	= 3000; VT = 1730
	lam	  = rho*( VL**2 - 2*(VT**2) ); mu= rho*( VT**2); 
	eta_v = 1e-10   # (Pa s) bulk viscosity
	eta_s = 1e-10   # (Pa s) shear viscosity
	material2 = Material(name=name,rho=rho,c11 = lam + 2*mu,c12=lam,c22=lam + 2*mu,c44=mu,eta_v= eta_v, eta_s=eta_s,label=60)
	#Append to the material list
	materials.append(material2)
	

	#Define Inspection Type
	#   TypeLaunch : 'PulseEcho'
	#   TypeLaunch : 'Transmission'
	#
	source	   = Source(TypeLaunch = 'Transmission')
	

	#Define Transducer Object
	# Size, Offset, BorderOffset in mm
	transducer = Transducer(Size = 0.2, Offset=0, BorderOffset=0, Location=0, name = 'emisor')
	

	#Define Input Source Signal
	#   name : "RaisedCosinePulse"   
	#   name : "RickerPulse"
	signal	   = Signal(Amplitude=1000, Frequency=frequency, name="RaisedCosinePulse") 

	#Setup Simulation Parameters
	#
	# Usage:
	#  i)   First Define an Instance of the SimulationModel Object
	#  ii)  execute the method class: jobParameters using as input the materials list
	#  iii) execute the method class: createNumerical Model using as input the scenario
	#  iv)  execute the method class: initReceivers to initialize the receivers
	#  v)   execute the mtehod class: save signal using as input the attribute simModel.t
	#  vi)  save the Device into the simModel.Device attribute

	simModel   = SimulationModel(TimeScale=0.5, MaxFreq=2.0*frequency, PointCycle=10, SimTime=Time, SpatialScale=SpatialScale)
	simModel.jobParameters(materials)
	simModel.createNumericalModel(image)
	simModel.initReceivers()
	signal.saveSignal(simModel.t)
	simModel.Device = DEVICE


	TimeIter = len(simModel.t)


	#Define Main EFIT2D Object
	if VISCO:
		FD = EFIT2D(image, materials, source, transducer, signal, simModel,"VISCOELASTIC", Local_Size)
		print "Visco-EFIT2D"
	else:
		FD = EFIT2D(image, materials, source, transducer, signal, simModel,"ELASTIC", Local_Size)
		print "Elastic-EFIT2D"
	
	#setup receiver line (100 receivers)
	y = np.linspace(1,100,100)
	x = np.zeros((np.size(y)))
	FD.ReceiverVectorSetup(x,y)
	
	start = time.time()

	if Plotting:

		Z				 =		FD.SV
		fig				 =		glumpy.figure((int(FD.MRI/4.0),int(FD.NRI/4.0)) )
		I				 =		glumpy.Image(Z, interpolation='bilinear', colormap= glumpy.colormap.IceAndFire,
														vmin=-40, vmax=0)


		@fig.event
		def on_key_press(key, modifiers):
			if key == glumpy.window.key.ESCAPE:
				sys.exit();
			else:
				pass

		@fig.event
		def on_draw():
			fig.clear()
			I.draw(x=0, y=0, z=0, width=fig.window.width, height=fig.window.height)


		@fig.event
		def on_idle(*args):

			if (FD.n < TimeIter):
				FD.Run()
				FD.n+=1

				if FD.n % 500==0:
					print str(FD.n)	 + " Total " + str(TimeIter)

				if (FD.n % 500==0):
				
					FD.RunGL()
					
					Z[...]	=  FD.SV
					I.update()
					fig.redraw()

					if EnableVideo:
						fileName = FILE + str(FD.n/100) + ".jpg"
						FD.save_video(fig,fileName)

			else:
				global start
				stopT = time.time()-start
				print "Total Computation Time: ", stopT
				
				start = time.time()
				#Retrieve the receiver signals from the computing device to the host
				FD.saveOutput() #FD.receivers_signals
				
				# Save simulation data: receivers, dt, dr, dz
				FD.save_data(FILE)
				FD.save_data_receivers(FILE + "_receivers")
				
				
				plt.figure();
				vector_t = np.arange(0,TimeIter)*FD.dt*1e6
				sig = FD.receiver_signals
				plt.plot(vector_t,sig)
				plt.show()
				
				sys.exit();


		glumpy.show()


	else:
		# main loop
		while (FD.n < TimeIter):
			FD.Run()
			FD.n+=1
			if FD.n % 500 == 0:
				print FD.n, " of total iterations: ", TimeIter
		

		stopT = time.time()-start
		print "Total Computation Time: ", stopT

		start = time.time()

		#Retrieve the receiver signals from the computing device to the host
		print "recovery"
		FD.saveOutput() #FD.receivers_signals

		# Save simulation data: receivers, dt, dr, dz
		FD.save_data(FILE)

		FD.save_data_receivers(FILE + "_receivers")

		# plot receiver
		vector_t = np.arange(0,TimeIter)*FD.dt
		plt.figure()
		sig = FD.receiver_signals
		plt.plot(vector_t,sig)
		
		plt.figure()
		sigs = FD.receiversX
		plt.plot(sigs)
		
		plt.show()
	
