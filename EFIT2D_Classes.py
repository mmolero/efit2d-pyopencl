#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package EFIT2D_Classes

Support Library: efit2d-pyopencl

Manuscript Title: Optimized OpenCL implementation of the Elastodynamic Finite Integration Technique for viscoelastic media

Authors: M Molero, U Iturraran-Viveros, S Aparicio, M.G. Hernández 

Program title: EFIT2D-PyOpenCL

Journal reference: Comput. Phys. Commun.

Programming language: Python.

External routines:  numpy, scipy, matplotlib, glumpy, pyopencl

Computer: computers having GPU or Multicore CPU with OpenCL drivers.


	All classes here defined are used to define:
		- The scenario,
		- Material objects,
		- Input sources,
		- Inspection setup,
		- Simulation parameters

"""

import		  numpy				  as      np
from		  math				  import  sin, cos, sqrt, pi, exp
import		  random
import		  time
from		  scipy				  import  signal
from		  scipy.fftpack		  import  fftshift
from skimage.transform import rotate

try:
	from	Image import		Image  
except:
	from    PIL import Image

from		  matplotlib		  import  cm
import		  matplotlib.pyplot	  as      plt


def imresize(arr, size, **kwargs):
    from PIL import Image
    size_list = [int(arr.shape[0] * size), int(arr.shape[1] * size)]
    return np.array(Image.fromarray(arr).resize(size_list))


def imrotate(arr, angle, **kwargs):
    return rotate(arr, angle=angle)


def RaisedCosinePulse(t, Freq, Amplitude):
	"""
	Raised-Cosine Pulse
	
	@param t   time vector
	@param Freq  Frequency in Hz
	@param Amplitude  Real Value of Amplitude
		
	@return  Output signal vector
	@retval  P vector of length equals to the time vector t
	
	""" 
	N  = np.size(t,0)
	P  = np.zeros((N,),dtype=np.float32)
	for m in range(0,N):
		if t[m] <= 2.0/Freq:
			P[m] = Amplitude *(1-cos(pi*Freq*t[m]))*cos(2*pi*Freq*t[m])

	return P



def	ricker(t,ts,fsavg):
	"""
	Ricker Pulse

	@param t      time vector
	@param ts     temporal delay
	@param fsavg  pulse width parameter

	@return  Output signal vector
	"""

	a  = fsavg*pi*(t-ts)
	a2 = a*a
	return ((1.0-2.0*a2)*np.exp(-a2))

##



class NewImage:
	"""
	Class NewImage: Definition of the Main Geometric Scenario.
	"""
	
	def __init__(self, Width=40, Height=40,Pixel_mm=10,label=0,SPML=False):
		"""
		Constructor of the Class NewImage

		@param Width      Width of the Scenario
		@param Height     Height of the Scenario
		@param Pixel_mm   Ratio Pixel per mm
		@param label      Label
		@param SPML       Flag used to indicate the boundary conditions


		"""
		## Width of the Scenario
		self.Width		 = Width
		## Height of the Scenario
		self.Height		 = Height
		## Ratio Pixel per mm
		self.Pixel_mm	 = Pixel_mm
		## Label
		self.Label   	 = label
			
		## Flag used to indicate the boundary conditions
		self.SPML		 = SPML

		## Dimension 1 of the Scenario Matrix
		self.M			 = int(self.Height * self.Pixel_mm)
		## Dimension 2 od the Scenario Matrix
		self.N			 = int(self.Width  * self.Pixel_mm)
		
		## Scenarion Matrix (MxN)
		self.I			 = np.ones((self.M,self.N),dtype=np.uint8)*label
		self.Itemp		 = 0
		
		## Size of the Boundary Layer
		self.Tap         = 0

		## Configure if boundary layers will be treated as absorbing layers or air layers.
		#
		#   False: Absorbing layers
		#
		#   True : Air boundaries
		self.AirBoundary = False


		
	def createLayer(self, centerW, centerH, Width, Height, label, Theta=0):
		"""
		Create a Layer

		@param  centerW   center in width-axis of the Layer
		@param  centerH   center in height-axis of the Layer
		@param  Width     Width of the Layer
		@param  Height    Height of the Layer
		@param  label     Label of the layer
		@param  Theta     Rotation Angle
		"""


		a  = int(Height*self.Pixel_mm/2.0) 
		b  = int(Width*self.Pixel_mm/2.0) 
		for	 x in  range(-a,a):
			for y in range(-b,b):
				tempX = round (x + centerH*self.Pixel_mm)
				tempY = round (y + centerW*self.Pixel_mm)
				self.I[tempX,tempY] = label

		if Theta != 0:
			self.I = imrotate(self.I,Theta,interp='nearest')

		
		
	def createABS(self,Tap):
		"""
		Create the boundary layers depending on the boundary conditions required

		@param Tap  Layer Size


		"""

		self.Tap		 = Tap
		self.SPML		 = True

		self.AirBoundary = False
		
		self.M, self.N   = np.shape(self.I)

		TP		     = round(Tap* self.Pixel_mm )
		M_pml		 = int( self.M	 + 2*TP )
		N_pml		 = int( self.N	 + 2*TP )
	
		self.Itemp		 = 255.0*np.ones((M_pml,N_pml),dtype=np.uint8)
		self.Itemp[TP : M_pml-TP, TP : N_pml-TP] = np.copy(self.I)

		

class Material:
	"""
	Class Material: Definition of a material

	@param name Material Name
	@param rho  Density (kg/m3)
	@param c11  C11 (Pa)
	@param c12  C12 (Pa)
	@param c22  C22 (Pa)
	@param c44  C44 (Pa)
	@param eta_v Bulk  Viscosity Constant (Pa s)
	@param eta_s Shear Viscosity Constant (Pa s)
	@param label Material Label

	"""
	def __init__(self, name="Water",rho=1000,c11=2.19e9,c12=0.0,c22=0.0,c44=0.0,eta_v=0, eta_s=0,label=0):
		
		"""
		Constructor of the Material object
		"""
		## Material Name
		self.name	=  name

		##Density (kg/m3)
		self.rho	=  rho

		## C11 (Pa)
		self.c11	=  c11

		## C12 (Pa)
		self.c12	=  c12

		## C22 (Pa)
		self.c22	=  c22

		## C44 (Pa)
		self.c44	=  c44

		## Longitudinal Velocity (m/s)
		self.VL		=  sqrt( c11/rho )

		## Shear Velocity (m/s)
		self.VT		=  sqrt( c44/rho )
		
		## Bulk  Viscosity Constant (Pa s)
		self.eta_v  =  eta_v
		
		## Shear Viscosity Constant (Pa s)
		self.eta_s  =  eta_s

		## Material Label
		self.Label	=  label
		
	def __str__(self):
			return "Material:" 

	def __repr__(self):
			return "Material:" 
		


class Source:
	"""
	Class Source: Define the Inspection Type

	@param TypeLaunch   Type of Inspection: Transmission or PulseEcho

	"""
	def __init__(self,TypeLaunch = 'Transmission'):

		##  Type of Inspection: Transmission or PulseEcho
		self.TypeLaunch		    = TypeLaunch

		## Define the location of the transducers in function of the type of the Inspection
		self.Theta			    = 0

		
		if	 self.TypeLaunch == 'PulseEcho':
			self.pulseEcho()
			
		elif self.TypeLaunch == 'Transmission':
			self.transmission()
		
	def __str__(self):
		return "Source: " 

	def __repr__(self):
		return "Source: "
		
		
	def pulseEcho(self):
		"""
		Define Theta for PulseEcho Inspection. PulseEcho Inspection uses the same transducer acting as emitter and as receiver
		"""
		self.Theta = [270*pi/180, 270*pi/180]
		

	def transmission(self):
		"""
		Define Theta for Transmission Inspection. Transmision uses two transducers, one used as emitter and another as receiver
		"""
		self.Theta = [270*pi/180, 90*pi/180]
		
		
		

class Transducer:
	"""
	Class Transducer:  Definition of the Transducer Object

	@param Size          Transducer Size
	@param Offset        Offset position of the Transducer. By default is set to zero
	@param BorderOffset  Border offset position of the Transducer. By default is set to zero
	@param Location      Location is set to zero that indicates Up location
	@param name          Transducer Name


	"""
	def __init__(self, Size = 10, Offset=0, BorderOffset=0, Location=0, name = 'emisor'):
		"""
		Constructor of the Class Transducer
		"""

		# Location = 0 => Top
		
		## Transducer Size
		self.Size		  = Size
		
		## Offset position of the Transducer. By default is set to zero
		#
		# This offset is measured taking into account the center of the Scenario in the width-axis
		#
		# Positive Values indicate offsets toward the right
		#
		# Negative values indicate offsets toward the left
		self.Offset		  = Offset
		
		## Border offset position of the Transducer. By default is set to zero
		#
		# This border offset takes into account the center od the Scenario in the width axis
		# but this offset is measured in direction of the height-axis
		#
		# Only Positive values must be defined.
		self.BorderOffset = BorderOffset
		
		##Size of the trasnducer in Pixels
		self.SizePixel	  = 0
		
		## Location-> 0: Top. This version only works when the location=0
		self.Location	  = Location
		
		## Name of the transducer
		self.name         = name
		
		
	
	def __str__(self):
		return "Transducer: " 

	def __repr__(self):
		return "Transducer: "

	
####################################################################################
	
		
class Signal:
	"""
	Class Signal: Signal Definition (Source Input for the Simulation)

	@param Amplitude   Signal Amplitude
	@param Frequency   Frequency Amplitude
	@param Name        Name of the Signal:  RaisedCosinePulse or RickerPulse
	@param ts          Time Delay: used only for RickerPulse


	"""
	def __init__(self, Amplitude=1, Frequency=1e6, name ="RaisedCosinePulse", ts=1):

		## Signal Amplitude
		self.Amplitude = Amplitude
		
		## Frequency Amplitude
		self.Frequency = Frequency
		
		## Name of the Signal:  RaisedCosinePulse or RickerPulse
		self.name      = name

		## Time Delay: used only for RickerPulse
		if ts == 1:		
			self.ts        = 3.0/Frequency;
		
		
	
	def __str__(self):
		return "Signal: " 

	def __repr__(self):
		return "Signal: "
		
		

	def generate(self,t):
		"""
		Generate the signal waveform

		@param t  vector time
		@return signal vector with the same length as the vector time

		"""

		if self.name == "RaisedCosinePulse":
			return RaisedCosinePulse(t, self.Frequency, self.Amplitude)
		elif self.name == "RickerPulse":
			return ricker(t, self.ts, self.Frequency)
			
	def saveSignal(self,t):	
		"""
		Save the signal waveform into the object
		@param t  vector time

		"""
		self.time_signal  = self.generate(t)
				




######################################
class Inspection:
	"""
	Class Inspection:  used for the configuration of the inspections to be emulated
	"""
	

	def __init__(self):
		"""
		Constructor of the Class Inspection
		"""

		## Position of the Transducer (Angle)
		self.Theta	= 0
		
		## Vector x-axis Position of the Transducer
		self.XL		= 0
		
		## Vector y-axis Position of the Transducer
		self.YL		= 0
		
		##
		self.IR		= 0
		

	def __str__(self):
		return "Inspection: " 

	def __repr__(self):
		return "Inspection: "
		
	

	def setTransmisor(self, source, transducer, x2, y2, X0, Y0):

		self.Theta	= source.Theta

		Ntheta		= np.size(self.Theta,0)
		NXL			= int(2*transducer.SizePixel)

		xL			= np.zeros((NXL,),dtype=np.float32)
		yL			= np.zeros((NXL,),dtype=np.float32)

		for m in range(0,Ntheta):

			if np.abs(np.cos(self.Theta[m])) < 1e-5:
				yL = np.linspace(y2[m]-transducer.SizePixel,y2[m]+transducer.SizePixel,num=NXL, endpoint=True)
				xL[:] = x2[m]*np.ones((NXL,),dtype=np.float32)


			elif np.abs(np.cos(self.Theta[m])) == 1:
				xL[:] = np.linspace(x2[m]-transducer.SizePixel, x2[m]+transducer.SizePixel,num=NXL, endpoint=True)
				yL[:] = y2[m] - ( (x2[m]-X0 )/( y2[m]-Y0 ) )*( xL[:]-x2[m] )

			else:
				xL[:] = np.linspace(x2[m]-(transducer.SizePixel*np.abs(np.cos(self.Theta[m]))),x2[m]+(transducer.SizePixel*np.abs(np.cos(self.Theta[m]))), num=NXL, endpoint=True )
				yL[:] = y2[m] - ( (x2[m]-X0 )/( y2[m]-Y0 )	)*( xL[:]-x2[m] )

			if m==0:
				self.XL		= np.zeros((np.size(xL,0),Ntheta),dtype=np.float32)
				self.YL		= np.zeros((np.size(xL,0),Ntheta),dtype=np.float32)


			self.XL[:,m]  = (np.around(xL[:]))
			self.YL[:,m]  = (np.around(yL[:]))
			


	def addOffset(self, image, transducer, NRI):
		"""
		Handle Offset

		"""
		NXL	   = np.size(self.XL,0)
		Ntheta = np.size(self.Theta,0)
		
		M_pml, N_pml = np.shape(image.Itemp)

		self.YL +=	 (np.around(transducer.Offset * image.Pixel_mm * NRI / float(N_pml)))

		self.IR		 = np.zeros((Ntheta,Ntheta),dtype=np.float32)
		B			 = list(range(0,Ntheta))
		self.IR[:,0] = np.int32(B[:])

		for i in range(1,Ntheta):
			B  = np.roll(B,-1)
			self.IR[:,i] = np.int32(B)
			
	def addBorderOffset(self, image, transducer, MRI):
		"""
		Handle  Border Offset

		"""

		M_pml, N_pml = np.shape(image.Itemp)
		ratio = float(MRI) / float(M_pml)
		
		self.XL[:,0] += (np.around(transducer.BorderOffset * image.Pixel_mm * ratio) )
		self.XL[:,1] -= (np.around(transducer.BorderOffset * image.Pixel_mm * ratio) )
		
	def flip(self):
		self.XL       = np.fliplr(self.XL)


	def SetReception(self,T):

		ReceptorX = (self.XL)
		ReceptorY = (self.YL)
		M,N		  = np.shape(ReceptorX)
		temp  = np.zeros((M,N-1),dtype=np.float32)
		
		for	 mm	 in range(0,M):
			for ir in  range(0,N-1):
				
				temp[mm,ir]	  =	 T[ int(ReceptorX[ mm,int(self.IR[0,ir+1]) ] ) , int(ReceptorY[ mm,int(self.IR[0,ir+1]) ]) ]
		
		if self.Field:
			return temp.transpose()
		else:
			return np.mean(temp,0)
		
		
	def SetReceptionVector(self, T, x, y):	
		M		  = np.size(x)
		temp      = np.zeros((M,),dtype=np.float32)
		for	 mm	in range(0,M):
			temp[mm] = T[(int(x[mm])),(int(y[mm]))]
		
		return temp	
		
		
			
		

class SimulationModel:
	"""
	Class  Simulation: setup the parameters for the  numerical simulation 

	Usage:
		- First Define an Instance of the SimulationModel Object
		- Execute the method class: jobParameters using as input the materials list
		- Execute the method class: createNumerical Model using as input the scenario
		- Execute the method class: initReceivers to initialize the receivers
		- Execute the mtehod class: save signal using as input the attribute simModel.t
		- Save the Device into the simModel.Device attribute


	@param TimeScale    Scale Time Factor
	@param MaxFreq      Maximum Frequency
	@param PointCycle   Points per Cycle
	@param SimTime      Time Simuation
	@param SpatialScale Spatial Scale: 1 -> meters, 1e-3 -> millimeters


	"""
	def __init__(self,TimeScale=1, MaxFreq=2e6, PointCycle=10, SimTime=50e6, SpatialScale=1e-3):

		## Scale Time Factor
		self.TimeScale	= TimeScale
		
		## Maximum Frequency
		self.MaxFreq	= MaxFreq	  # MHz
		
		## Points per Cycle
		self.PointCycle = PointCycle
		
		## Time Simuation
		self.SimTime	= SimTime	  # microseconds

		## Spatial Scale: 1 -> meters, 1e-3 -> millimeters
		self.SpatialScale = SpatialScale
		
		## Spatial Discretization
		self.dx         = 0

		## Temporal Discretization
		self.dt         = 0
		
		self.Rgrid      = 0
		self.TapG       = 0
		self.t          = 0
		self.Ntiempo    = 0
		
		self.MRI,self.NRI = (0,0)
		
		self.receiver_signals   = 0
		self.Device           = 'CPU'
		
		self.XL         = 0
		self.YL         = 0
		
		
	def __str__(self):
		return "Simulation Model: " 

	def __repr__(self):
		return "Simulation Model: "
		
	def jobParameters(self,materiales):
		"""
		Define Main Simulation Parameters

		@parm materiales  Materials List

		
		"""
		indVL = [mat.VL for mat in materiales if mat.VL > 400]
		indVT = [mat.VT for mat in materiales if mat.VT > 400]
							
				
		VL	  = np.array(indVL)
		VT	  = np.array(indVT)
		V	  = np.hstack( (VL, VT) )
		
		self.dx = np.float32( np.min([V]) / (self.PointCycle*self.MaxFreq) )
		self.dt = self.TimeScale * np.float32( 0.7071 * self.dx / (	 np.max([V]) ) )

			
		self.Ntiempo = int(round(self.SimTime/self.dt))
		self.t	= self.dt*np.arange(0,self.Ntiempo)
	
	
	def createNumericalModel(self, image):
		"""
		Create the Numerical Model

		@param image  The Scenario Object
		"""

		#Spatial Scale
		Mp			      =	 np.shape(image.Itemp)[0]*self.SpatialScale/image.Pixel_mm/self.dx
		self.Rgrid	      =	 Mp/np.shape(image.Itemp)[0]
		
		self.TapG	      =	 np.around(image.Tap * self.Rgrid * image.Pixel_mm)
		self.Im		      =	 imresize(image.Itemp, self.Rgrid, interp='nearest')
		self.MRI,self.NRI =	 np.shape(self.Im)
	
		print("dt: " + str(self.dt) + " dx: "  + str(self.dx) + " Grid: " +  str(self.MRI) + " x " + str(self.NRI))
	
		
	def initReceivers(self):
		"""
		Initialize the receivers

		"""
		self.receiver_signals = 0
	
	
	def setDevice(self,Device):
		"""
		Set the Computation Device

		@param Device Device to be used

		Define the device used to compute the simulations:
			 - "CPU"        : uses the global memory in th CPU
			 - "GPU_Global" : uses the global memory in the GPU
			 - "GPU_Local"  : uses the local memory in the GPU

		"""

		if Device == 0:
			self.Device = 'CPU'
		elif Device ==1:
			self.Device = 'GPU_Global'
		elif Device ==2: 
			self.Device = 'GPU_Local'
			
		
		
	

		



