#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package EFIT2D

Class EFIT2D


Manuscript Title: Optimized OpenCL implementation of the Elastodynamic Finite Integration Technique for viscoelastic media

Authors: M Molero, U Iturraran-Viveros, S Aparicio, M.G. Hernández 

Program title: EFIT2D-PyOpenCL

Journal reference: Comput. Phys. Commun.

Programming language: Python.

External routines:  numpy, scipy, matplotlib, glumpy, pyopencl

Computer: computers having GPU or Multicore CPU with OpenCL drivers.

"""


from	EFIT2D_Classes	   import		*
import	numpy			   as			np
import	pyopencl		   as			cl

try:
	from	Image import		Image  
except:
	from    PIL import Image
	
	
import	OpenGL.GL as	   GL
import	copy

from	scipy.io import savemat
import  time



class EFIT2D:
	"""
	Class EFIT2D:  Main Object to perform the EFIT2D technique

	@param Image       Scenario Object
	@param Materials   Materials List
	@param Source      Source Object
	@param Transducer  Transducer Object
	@param Signal      Signal Object
	@param SimModel    SimulationModel Object
	@param TypeSim     Simulation Type: ELASTIC or VISCO
	@param DimLocal    Local Size


	"""


	def __init__(self, Image, Materials, Source, Transducer, Signal, SimModel, TypeSim="ELASTIC", DimLocal=(16,16)):

		##  Numerical Scenario Matrix (MRI, NRI)
		self.Im					  = np.float32(np.copy(SimModel.Im))
		
		## Dimension of of the Numerical Scenario Matrix
		self.MRI, self.NRI		  = np.shape(self.Im)
		
		self.M_abs, self.N_abs	  = np.shape(Image.Itemp)
		
		self.Theta				  = Source.Theta
		
		## Time Discretization
		self.dt					  = SimModel.dt
		
		## Spatial Discretization
		self.dx					  = np.float32(SimModel.dx)
		
		self.Frequency			  = Signal.Frequency
		
		self.StopSignal			  = np.around( (2.0/self.Frequency)*(1/SimModel.dt) )
		
		self.SimDevice			  = SimModel.Device
		
		self.TypeSim			  = TypeSim
		

		self.DimLocalX, self.DimLocalY = DimLocal


		if SimModel.Device == 'GPU_Global' or SimModel.Device=='GPU_Local':
			self.Device = 'GPU'
		else:
			self.Device = 'CPU'

		Materiales = copy.deepcopy(Materials)

		self.InitCL(self.Device)
		self.MaterialSetup(Materiales)
		self.Init_Fields(Signal, SimModel)
		self.ReceiverSetup(Image, Source, Transducer, SimModel)
		self.StaggeredProp()
		self.applyABS(Materiales, SimModel)
		self.n	= 0
		
		if Image.AirBoundary:
			self.ConfigAirBoundary()

		self.Init_Fields_CL(SimModel)
		
		self.time_v=[]
		self.time_t=[]
		self.time  =[]
		
		self.EnableReceivers = False	
			
	

	def InitCL(self, DEVICE="GPU"):

		"""
		Init CL Configuration

		@param DEVICE Set the device to be used

		"""

		try:
			for platform in cl.get_platforms():
				for device in platform.get_devices():
					if cl.device_type.to_string(device.type)== DEVICE:
						my_device =				 device
						print my_device.name, "	 ", cl.device_type.to_string(my_device.type)

		except:
			my_device = cl.get_platforms()[0].get_devices()
			print my_device.name, "	 ", cl.device_type.to_string(my_device.type)

		self.ctx    = cl.Context([my_device])
		self.queue  = cl.CommandQueue(self.ctx)
		self.mf	    = cl.mem_flags

	

	def MaterialSetup(self, Materials):
		"""
		Material Setup
		@param Materials Materials List

		"""
		NumeroMat  = len(Materials)
		#Vacuum condition if some material is air
		for n in range(0,NumeroMat):
			if				Materials[n].rho < 2.0:
				Materials[n].rho = 10e23
				Materials[n].c11 = 1e-20
				Materials[n].c12 = 1e-20
				Materials[n].c22 = 1e-20
				Materials[n].c44 = 1e-20
				Materials[n].c44 = 1e-20
				Materials[n].c44 = 1e-20
				Materials[n].c44 = 1e-20
				Materials[n].eta_v = 1e-20
				Materials[n].eta_s = 1e-20

		self.rho				= np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.c11				= np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.c12				= np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.c22				= np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.c44				= np.ones((self.MRI,self.NRI) ,dtype=np.float32)*1e-30
		self.eta_v				= np.ones((self.MRI,self.NRI) ,dtype=np.float32)*1e-30
		self.eta_s				= np.ones((self.MRI,self.NRI) ,dtype=np.float32)*1e-30

		for i in range(0,self.MRI):
			for n in range(0,NumeroMat):
				ind =  np.nonzero(self.Im[i,:] == Materials[n].Label)
				self.rho[i,ind]	  = Materials[n].rho
				self.c11[i,ind]	  = Materials[n].c11
				self.c12[i,ind]	  = Materials[n].c12
				self.c22[i,ind]	  = Materials[n].c22
				self.c44[i,ind]	  = Materials[n].c44
				self.eta_v[i,ind] = Materials[n].eta_v
				self.eta_s[i,ind] = Materials[n].eta_s
				if self.c44[i,ind].any() == 0:
					self.c44[i,ind]	  = 1e-30
					self.eta_v[i,ind] = 1e-30
					self.eta_s[i,ind] = 1e-30


		for i in range(0,self.MRI):
			ind = np.nonzero(self.Im[i,:] == 255.0)
			self.rho[i,ind]	  = Materials[0].rho
			self.c11[i,ind]	  = Materials[0].c11
			self.c12[i,ind]	  = Materials[0].c12
			self.c22[i,ind]	  = Materials[0].c22
			self.c44[i,ind]	  = Materials[0].c44
			self.eta_v[i,ind] = Materials[0].eta_v
			self.eta_s[i,ind] = Materials[0].eta_s
			if self.c44[i,ind].any() == 0:
				self.c44[i,ind]	  = 1e-30
				self.eta_v[i,ind] = 1e-30
				self.eta_s[i,ind] = 1e-30


	def StaggeredProp(self):
		"""
		Configure the Staggered Grid
		"""

		BXtemp					 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		BYtemp					 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.BX					 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.BY					 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.C11				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.C12				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.C22				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.C44				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.ETA_V				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.ETA_VS				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.ETA_S				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.ETA_SS				 = np.zeros((self.MRI,self.NRI),dtype=np.float32)


		BXtemp[:,:]				  =		   1.0/self.rho[:,:]
		BYtemp[:,:]				  =		   1.0/self.rho[:,:]

		self.C11				   =	   np.copy(self.c11)
		self.C12				   =	   np.copy(self.c12)
		self.C22				   =	   np.copy(self.c22)

		self.ETA_V				   =	   np.copy(self.eta_v)	   
		self.ETA_S				   =	   np.copy(self.eta_s)
		self.ETA_VS				   =       self.ETA_V + 2*self.ETA_S

		self.BX[:-2,:]	   =  0.5*( BXtemp[1:-1,:] + BXtemp[:-2,:] )
		self.BX[ -2,:]	   =  np.copy(BXtemp[-2,:])

		self.BY[:,:-2]	   =  0.5*( BYtemp[:,1:-1]				 + BYtemp[:,:-2]  )
		self.BY[:, -2]	   =  np.copy(BYtemp[:,-2])

		self.C44[:-2,:-2]		  = 4./(  (1./self.c44[:-2,:-2] ) +		 (1./self.c44[1:-1,:-2]) +		(1./self.c44[:-2,1:-1] ) +				(1./self.c44[1:-1,1:-1] )		   )
		self.ETA_SS[:-2,:-2]  = 4./(  (1./self.eta_s[:-2,:-2] ) +  (1./self.eta_s[1:-1,:-2]) +	(1./self.eta_s[:-2,1:-1] ) +	  (1./self.eta_s[1:-1,1:-1] )	   )


	def ConfigAirBoundary(self):
		"""
		Configure the
		"""

		indx,indy = np.nonzero(self.Im == 255)
		self.BX[indx,indy]				 = 0.0
		self.BY[indx,indy]				 = 0.0
		self.C11[indx,indy]				 = 0.0
		self.C12[indx,indy]				 = 0.0
		self.C22[indx,indy]				 = 0.0
		self.C44[indx,indy]				 = 0.0


	def setInspection(self,Image,Source, Transducer, SimModel):

		insp = Inspection()

		D_T = (self.MRI-1.)/2.
		x2				= self.MRI/2.  + (D_T - SimModel.TapG)*np.sin(Source.Theta)
		y2				= self.NRI/2.  + (D_T - SimModel.TapG)*np.cos(Source.Theta)

		X0				= self.MRI/2.
		Y0				= self.NRI/2.

		Transducer.SizePixel =	np.around( 0.5 * Image.Pixel_mm * Transducer.Size * float(self.NRI) / self.N_abs )

		insp.setTransmisor(Source,Transducer,x2,y2,X0,Y0)
		insp.addOffset(Image, Transducer, self.NRI)
		insp.addBorderOffset(Image, Transducer, self.MRI)

		return insp


	def ConfigFuente(self, Image, Source, Transducer, SimModel):

		Trans = Transducer
		self.insp		  = self.setInspection(Image,Source, Trans, SimModel)
		self.XL			  = np.copy(self.insp.XL)
		self.YL			  = np.copy(self.insp.YL)
		self.IR			  = np.copy(self.insp.IR)
		self.Ratio        = Image.Pixel_mm*SimModel.Rgrid


	def applyABS(self,Materials, SimModel):

		APARA = 0.015
		for i in range(0,self.MRI):
			for j in range(0,self.NRI):
				if	 i < SimModel.TapG:
					self.ABS[i,j] = np.exp(-((APARA*(SimModel.TapG-i))**2))
				elif j < SimModel.TapG:
					self.ABS[i,j] = np.exp(-((APARA*(SimModel.TapG-j))**2))
				elif i > (self.MRI-SimModel.TapG+1):
					self.ABS[i,j] = np.exp(-((APARA*(i-self.MRI+SimModel.TapG-1))**2))
				elif j > (self.NRI-SimModel.TapG+1):
					self.ABS[i,j] = np.exp(-((APARA*(j-self.NRI+SimModel.TapG-1))**2))
				else:
					self.ABS[i,j] = 1.0



	def Init_Fields(self, Signal, SimModel):

		self.input_source  = Signal.generate(SimModel.t)
		self.vx	 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.vy	 = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.dvx = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.dvy = np.zeros((self.MRI,self.NRI),dtype=np.float32)

		self.Txx  = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.Txy  = np.zeros((self.MRI,self.NRI),dtype=np.float32)
		self.Tyy  = np.zeros((self.MRI,self.NRI),dtype=np.float32)

		self.Gxx  = np.zeros( (self.MRI,self.NRI), dtype=np.float32)
		self.SV	  = np.zeros( (self.MRI,self.NRI), dtype=np.float32)
		self.ABS  = np.zeros( (self.MRI,self.NRI), dtype=np.float32)



	def Init_Fields_CL(self, SimModel):


		self.receiver_buf  = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.receiver_signals)
		
		self.Txx_buf	   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.Txx)
		self.Tyy_buf	   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.Tyy)
		self.Txy_buf	   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.Txy)
		self.vx_buf		   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.vx)
		self.vy_buf		   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.vy)
		self.dvx_buf	   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.dvx)
		self.dvy_buf	   = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.dvy)
		

		self.ConfigSource()

		self.ABS_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.ABS)
		self.BX_buf		   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.BX)
		self.BY_buf		   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.BY)
		self.C11_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.C11)
		self.C12_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.C12)
		self.C44_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.C44)
		self.ETA_VS_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.ETA_VS)
		self.ETA_S_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.ETA_S)
		self.ETA_SS_buf	   = cl.Buffer(self.ctx, self.mf.READ_ONLY	| self.mf.COPY_HOST_PTR, hostbuf=self.ETA_SS)

		
		self.NX	  = np.size(self.XL,0)
		self.XLL  = np.copy(np.int32(self.XL[:,0]))
		self.YLL  = np.copy(np.int32(self.YL[:,0]))
		self.XXL  = np.copy(np.int32(self.XL[:,1]))
		self.YYL  = np.copy(np.int32(self.YL[:,1]))

		self.XL_buf	  = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf=self.XLL)
		self.YL_buf	  = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf=self.YLL)
		self.XXL_buf  = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf=self.XXL)
		self.YYL_buf  = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf=self.YYL)
		
		
		self.dtx	 = np.float32(SimModel.dt/SimModel.dx)
		self.dtdxx	 = np.float32(SimModel.dt/(SimModel.dx))


		def RoundUp(groupSize, globalSize):
			r = globalSize % groupSize;
			if r == 0:
				return globalSize;
			else:
				return globalSize + groupSize - r;

		self.globalWorkSize = (RoundUp(self.DimLocalY, self.NRI), RoundUp(self.DimLocalX, self.MRI ))
		self.program			= cl.Program(self.ctx, self.EFIT2D_Kernel() ).build()
		

	def ConfigSource(self):
		NX	= np.size(self.XL,0)
		
		for IT in range(-3,0):
			for m in range(0,NX):

				xl				= int(self.XL[m,0])
				yl				= int(self.YL[m,0])

				self.BX[xl+IT,yl] = 0.0
				self.BY[xl+IT,yl] = 0.0
				self.C11[xl+IT,yl] = 0.0
				self.C12[xl+IT,yl] = 0.0
				self.C22[xl+IT,yl] = 0.0
				self.C44[xl+IT,yl] = 0.0
				self.ETA_VS[xl+IT,yl] = 0.0
				self.ETA_S[xl+IT,yl] = 0.0
				self.ETA_SS[xl+IT,yl] = 0.0


	def ReceiverSetup(self, Image, Source,Transducer, SimModel):

		self.ConfigFuente(Image,Source,Transducer,SimModel)
		self.receiver_signals = np.zeros(( SimModel.Ntiempo, np.size(self.IR,1)-1 ),dtype=np.float32)
		self.Ntiempo = SimModel.Ntiempo
		self.TapG = SimModel.TapG
	
	
	def ReceiverVectorSetup(self, x, y):
		self.EnableReceivers = True
		self.Rx = np.around(x*self.Ratio) + self.TapG +1
		self.Ry = np.around(y*self.Ratio) + self.TapG +1
		self.receiversX = np.zeros(( self.Ntiempo, np.size(x)  ),dtype=np.float32)
		self.receiversY = np.zeros(( self.Ntiempo, np.size(x)  ),dtype=np.float32)
		
	
	def getReceivers(self, T1,T1_buf, T2,T2_buf):
		cl.enqueue_copy(self.queue, T1, T1_buf)
		cl.enqueue_copy(self.queue, T2, T2_buf)
		self.receiversX[self.n,:] = self.insp.SetReceptionVector(T1, self.Rx, self.Ry)
		self.receiversY[self.n,:] = self.insp.SetReceptionVector(T2, self.Rx, self.Ry)	
		

	
	def save_data_receivers(self, File):
		if self.EnableReceivers:
			data = {}
			data['receiversX'] = np.copy(self.receiversX)
			data['receiversY'] = np.copy(self.receiversY)
			data['dt']		  = np.copy(self.dt)
			savemat(File,data)
		
		
	def saveOutput(self):	
		cl.enqueue_copy(self.queue, self.receiver_signals, self.receiver_buf).wait()
		
	def save_data(self, File):

		data = {}
		data['receiver'] = np.copy(self.receiver_signals)
		data['dt']		 = np.copy(self.dt)
		savemat(File,data)


	def save_video(self, fig, File):
		"""
		Capture image from the OpenGL buffer
		@param fig   Figure
		@param  File  Filename

		"""
		buffer = (GL.GLubyte * (3*fig.window.width*fig.window.height) )(0)
		GL.glReadPixels(0, 0, fig.window.width, fig.window.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buffer)

		# Use PIL to convert raw RGB buffer and flip the right way up
		image = Image.fromstring(mode="RGB", size=(fig.window.width, fig.window.height), data=buffer)
		image = image.transpose(Image.FLIP_TOP_BOTTOM)
		image.save(File)


	def Run(self):

		if self.TypeSim=="ELASTIC":
			self.RunCL()
		if self.TypeSim=="VISCOELASTIC":
			self.RunCL_Voigt()

	def RunCL(self):
		if self.SimDevice=='CPU':
			self.Run_Global()

		elif self.SimDevice=='GPU_Global':
			self.Run_Global()
		elif self.SimDevice=='GPU_Local':
			self.Run_Local()
	
		if self.EnableReceivers:
			self.getReceivers(self.Txx,self.Txx_buf,self.Tyy,self.Tyy_buf)
			


	def RunCL_Voigt(self):

		if self.SimDevice=='CPU':
			self.Run_Global_Voigt()

		elif self.SimDevice=='GPU_Global':
			self.Run_Global_Voigt()

		elif self.SimDevice=='GPU_Local':
			self.Run_Local_Voigt()
		
		if self.EnableReceivers:
			self.getReceivers(self.Txx,self.Txx_buf,self.Tyy,self.Tyy_buf)
		

	def Run_Global(self):

		start1 = time.time()
		self.program.Velocity_EFIT2D(self.queue, (self.NRI,self.MRI,), None,
											 self.Txx_buf, self.Txy_buf, self.Tyy_buf,
											 self.vx_buf,  self.vy_buf,	 self.BX_buf, self.BY_buf, self.ABS_buf).wait()
	
		stop = time.time()
		self.time_v.append(stop-start1)


		start = time.time()
		self.program.Stress_EFIT2D(self.queue, (self.NRI,self.MRI,), None,
										  self.Txx_buf, self.Txy_buf, self.Tyy_buf,
										  self.vx_buf,	self.vy_buf, self.C11_buf, self.C12_buf, self.C44_buf, self.ABS_buf).wait()
		
	
		
		stop = time.time()
		self.time_t.append(stop-start)


		y  = np.float32(self.input_source[self.n])*self.dtdxx

		source = self.program.Source_EFIT2D( self.queue, (self.NX,), None, self.Txx_buf, self.Tyy_buf, self.XL_buf, self.YL_buf, y).wait()					
		
		
	
		receiver = self.program.Receiver_EFIT2D( self.queue, (self.NX,), None,self.Txx_buf, self.receiver_buf, np.int32(self.n),
												 self.XXL_buf, self.YYL_buf).wait()
			
		
			
		self.time.append(time.time()-start1)
		
		

	def Run_Global_Voigt(self):

		start1 = time.time()
		self.program.Velocity_EFIT2D_Voigt(self.queue, (self.NRI,self.MRI,), None,
														  self.Txx_buf, self.Txy_buf, self.Tyy_buf,
														  self.vx_buf,	self.vy_buf,  self.dvx_buf, self.dvy_buf,
														  self.BX_buf,	self.BY_buf,  self.ABS_buf).wait()
		
		
		stop = time.time()
		self.time_v.append(stop-start1)

		start = time.time()
		self.program.Stress_EFIT2D_Voigt(self.queue, (self.NRI,self.MRI,), None,
											self.Txx_buf, self.Txy_buf, self.Tyy_buf,
											self.vx_buf,  self.vy_buf,	self.dvx_buf, self.dvy_buf,
											self.C11_buf, self.C12_buf, self.C44_buf,
											self.ETA_VS_buf, self.ETA_S_buf, self.ETA_SS_buf, self.ABS_buf).wait()
	
		stop = time.time()
		self.time_t.append(stop-start)

		y  = np.float32(self.input_source[self.n])*self.dtdxx

		self.program.Source_EFIT2D( self.queue, (self.NX,), None, self.Txx_buf, self.Tyy_buf, self.XL_buf, self.YL_buf, y).wait()														
	
		self.program.Receiver_EFIT2D( self.queue, (self.NX,), None,self.Txx_buf, self.receiver_buf, np.int32(self.n),
												 self.XXL_buf, self.YYL_buf).wait()
		
		self.time.append(time.time()-start1)
		
		
		
		
		
	def Run_Local(self):

		start1 = time.time()
		vel =	self.program.Velocity_Local(self.queue, self.globalWorkSize, (self.DimLocalY,self.DimLocalX),
											self.Txx_buf, self.Txy_buf, self.Tyy_buf,
											self.vx_buf,  self.vy_buf,	self.BX_buf, self.BY_buf, self.ABS_buf).wait()

		stop = time.time()
		self.time_v.append(stop-start1)


		start = time.time()
		stress = self.program.Stress_Local(self.queue, self.globalWorkSize, (self.DimLocalY,self.DimLocalX),
										   self.Txx_buf, self.Txy_buf, self.Tyy_buf,
										   self.vx_buf,	 self.vy_buf,
										   self.C11_buf, self.C12_buf, self.C44_buf, self.ABS_buf).wait()

		stop = time.time()
		self.time_t.append(stop-start)


	
		y  = np.float32(self.input_source[self.n])*self.dtdxx

		source = self.program.Source_EFIT2D( self.queue, (self.NX,), None, self.Txx_buf, self.Tyy_buf, self.XL_buf, self.YL_buf, y).wait()														
	
		receiver = self.program.Receiver_EFIT2D( self.queue, (self.NX,), None,self.Txx_buf, self.receiver_buf, np.int32(self.n),
												 self.XXL_buf, self.YYL_buf).wait()
			
		self.time.append(time.time()-start1)


	def Run_Local_Voigt(self):

		start1 = time.time()
		vel = self.program.Velocity_Local_Voigt(self.queue, self.globalWorkSize, (self.DimLocalY,self.DimLocalX),
										  self.Txx_buf, self.Txy_buf, self.Tyy_buf,
										  self.vx_buf, self.vy_buf, self.dvx_buf, self.dvy_buf,
										  self.BX_buf, self.BY_buf, self.ABS_buf).wait()

		stop = time.time()
		self.time_v.append(stop-start1)


		start = time.time()
		stress = self.program.Stress_Local_Voigt(self.queue, self.globalWorkSize, (self.DimLocalY,self.DimLocalX),
										self.Txx_buf, self.Txy_buf, self.Tyy_buf,
										self.vx_buf,  self.vy_buf,	self.dvx_buf, self.dvy_buf,
										self.C11_buf, self.C12_buf, self.C44_buf,
										self.ETA_VS_buf, self.ETA_S_buf, self.ETA_SS_buf, self.ABS_buf).wait()

		stop = time.time()
		self.time_t.append(stop-start)

		y  = np.float32(self.input_source[self.n])*self.dtdxx

		source = self.program.Source_EFIT2D( self.queue, (self.NX,), None, self.Txx_buf, self.Tyy_buf, self.XL_buf, self.YL_buf, y).wait()														
	
		receiver = self.program.Receiver_EFIT2D( self.queue, (self.NX,), None,self.Txx_buf, self.receiver_buf, np.int32(self.n),
												 self.XXL_buf, self.YYL_buf).wait()
			
		
		self.time.append(time.time()-start1)


	def RunGL(self, step=50):

		if self.n % step==0:
			cl.enqueue_copy(self.queue, self.vx, self.vx_buf)
			cl.enqueue_copy(self.queue, self.vy, self.vy_buf)
			self.SV	 = np.sqrt(self.vx**2 + self.vy**2 )
			self.SV	 = 20.*np.log10((np.abs(self.SV)/np.max(np.abs(self.SV+1e-40))) + 1e-40)



	def EFIT2D_Kernel(self):

		macro	 = """
					 #define MRI		%s
					 #define NRI		%s
					 #define ind(i, j)	( ( (i)*NRI) + (j) )
					 #define dtx		%gf
					 #define dtdxx		%gf
					 #define DimLocalX	%s
					 #define DimLocalY	%s
					 #define Stencil	2
					 #define NX			%s
					 #define dt		    %gf
					 #define ddx		    %gf



		"""%( str(self.MRI), str(self.NRI), self.dtx, self.dtdxx, 
			  str(self.DimLocalX), str(self.DimLocalY), str(self.NX), self.dt, np.float32(1.0/self.dx) )
		
		
		if self.TypeSim=="ELASTIC":
			f	 = open("EFIT2D.cl",'r')
		if self.TypeSim=="VISCOELASTIC":
			f	 = open("EFIT2D-VISCO.cl",'r')
		
		fstr = "".join(f.readlines())
		kernel_source = macro + fstr
		
		
		return kernel_source
		
