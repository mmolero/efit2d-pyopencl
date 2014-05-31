
/** Source Input Kernel
*/
__kernel void Source_EFIT2D( __global float *Txx,__global float *Tyy, __global const int *XL, __global const int *YL, const float source){

			  uint i =	get_global_id(0);
			  Txx[ind(XL[i],YL[i])] -= source;
		  	  Tyy[ind(XL[i],YL[i])] -= source;
		
}

/** Receiver Kernel
*/
__kernel void Receiver_EFIT2D( __global float *Txx, __global float *receiver, const int t,
							   __global const int *XXL, __global const int *YYL){
											
		
			   float _tmp = 0.0f;
			  
			   for (int i=0; i<get_global_size(0); ++i)
		       {    	 
				 _tmp +=  Txx[ind(XXL[i],YYL[i])];
			   }
				
			   
			   receiver[t] = _tmp/(float)get_global_size(0);
			   
			  
}


/** Velocity Kernel (Global Memory) for Elastic Case
*/
__kernel void Velocity_EFIT2D(__global float *Txx, __global float *Txy, __global float *Tyy,
							  __global float *vx,  __global float *vy,
							  __global const float *BX,	__global const float *BY,
							  __global const float *ABS){

		int j = get_global_id(0);
		int i = get_global_id(1);
	

	    if (i<MRI-1 && j>0){
			vx[ind(i,j)]  += (BX[ind(i,j)]*dtx)*( Txx[ind(i+1,j)] - Txx[ind(i,j)]	 + Txy[ind(i,j)]   - Txy[ind(i,j-1)] );
	    }

	    if (i>0 &&  j<NRI-1){
			vy[ind(i,j)] += (BY[ind(i,j)]*dtx)*( Txy[ind(i,j)]   - Txy[ind(i-1,j)] + Tyy[ind(i,j+1)] - Tyy[ind(i,j)]		 );
		
	    }
 

	    barrier(CLK_GLOBAL_MEM_FENCE);
		// Apply absorbing boundary conditions

		vx[ind(i,j)]	  *= ABS[ind(i,j)];
 	   	vy[ind(i,j)]	  *= ABS[ind(i,j)];
	 }


/** Stress Kernel (Global Memory) for Elastic Case
*/
__kernel void Stress_EFIT2D(__global float *Txx, __global float *Txy, __global float *Tyy,
							__global float *vx,	 __global float *vy,
							__global const float *C11,  __global const float *C12, __global const float *C44,
							__global const float *ABS) {


		int j = get_global_id(0);
 		int i = get_global_id(1);


 		if (i>0 && j>0 ){
				 Txx[ind(i,j)] += ( ( C11[ind(i,j)]* dtx )*(vx[ind(i,j)] - vx[ind(i-1,j)]) +
									( C12[ind(i,j)]* dtx )*(vy[ind(i,j)] - vy[ind(i,j-1)]) );

				 Tyy[ind(i,j)] += (	( C12[ind(i,j)]* dtx )*(vx[ind(i,j)] - vx[ind(i-1,j)]) +
									( C11[ind(i,j)]* dtx )*(vy[ind(i,j)] - vy[ind(i,j-1)]) );
       	}

 		if (i<MRI-1  && j<NRI-1){
				 Txy[ind(i,j)] +=	( C44[ind(i,j)]	 * dtx )*(vx[ind(i,j+1)] - vx[ind(i,j)] + vy[ind(i+1,j)] - vy[ind(i,j)] ) ;
 		}

 		barrier(CLK_GLOBAL_MEM_FENCE);

 		Txx[ind(i,j)]	*= ABS[ind(i,j)];
 		Tyy[ind(i,j)]	*= ABS[ind(i,j)];
 		Txy[ind(i,j)]	*= ABS[ind(i,j)];


}


/** Velocity Kernel (Local Memory) for Elastic Case
*/

__kernel void Velocity_Local(__global float *Txx, __global float *Txy, __global float *Tyy,
						     __global float *vx,  __global float *vy,
							 __global const float *BX, __global const float *BY, __global const float *ABS){


	 __local float local_Txx[DimLocalX+Stencil][DimLocalY+Stencil];
	 __local float local_Txy[DimLocalX+Stencil][DimLocalY+Stencil];
	 __local float local_Tyy[DimLocalX+Stencil][DimLocalY+Stencil];


	 uint halo = Stencil/2;
	 uint j	   = get_group_id(0) * get_local_size(0) + get_local_id(0);
	 uint i	   = get_group_id(1) * get_local_size(1) + get_local_id(1);

	 uint iBlock   = get_local_id(1)+halo;
	 uint jBlock   = get_local_id(0)+halo;

	 uint iField   = i + halo;
	 uint jField   = j + halo;

	 float D_Txx  = 0.0f;
	 float Dx_Txy = 0.0f;
	 float Dy_Txy = 0.0f;
	 float D_Tyy  = 0.0f;
	
	 float tabs, tdvx, tdvy, tvx, tvy;
	 

     if (i <  MRI - Stencil && j < NRI - Stencil ){


      		   barrier(CLK_LOCAL_MEM_FENCE);

			   // fill left halo of local memory
			   if (get_local_id(0) < halo )
			   {
							 local_Txx[iBlock][jBlock-halo] = Txx[iField*NRI + jField - halo];
							 local_Tyy[iBlock][jBlock-halo] = Tyy[iField*NRI + jField - halo];
							 local_Txy[iBlock][jBlock-halo] = Txy[iField*NRI + jField - halo];
						
			   }

			   // fill right halo of local memory
			   if (get_local_id(0) > DimLocalY - halo -1 )
			   {
							 local_Txx[iBlock][jBlock+halo] = Txx[iField*NRI + jField + halo];
							 local_Tyy[iBlock][jBlock+halo] = Tyy[iField*NRI + jField + halo];
							 local_Txy[iBlock][jBlock+halo] = Txy[iField*NRI + jField + halo];
							
			   }

			   // fill bottom halo of local memory
			   if (get_local_id(1) < halo )
			   {
							 local_Txx[iBlock-halo][jBlock] = Txx[(iField-halo)*NRI + jField];
							 local_Tyy[iBlock-halo][jBlock] = Tyy[(iField-halo)*NRI + jField];
							 local_Txy[iBlock-halo][jBlock] = Txy[(iField-halo)*NRI + jField];
							
			   }


			   // fill top halo of local memory
			   if (get_local_id(1) > DimLocalX - halo -1 )
			   {
							 local_Txx[iBlock+halo][jBlock] = Txx[(iField+halo)*NRI + jField];
							 local_Tyy[iBlock+halo][jBlock] = Tyy[(iField+halo)*NRI + jField];
							 local_Txy[iBlock+halo][jBlock] = Txy[(iField+halo)*NRI + jField];
							
			   }

			   local_Txx[iBlock][jBlock]		= Txx[iField*NRI + jField];
			   local_Tyy[iBlock][jBlock]		= Tyy[iField*NRI + jField];
			   local_Txy[iBlock][jBlock]		= Txy[iField*NRI + jField];
			  

                barrier(CLK_LOCAL_MEM_FENCE);

				D_Txx  = 0.0f;
				Dx_Txy = 0.0f;
				Dy_Txy = 0.0f;
				D_Tyy  = 0.0f;

				if (j!=0){
				    Dy_Txy = local_Txy[iBlock][jBlock] - local_Txy[iBlock][jBlock-1];
				}

				if (i!=0){
				    Dx_Txy = local_Txy[iBlock][jBlock] - local_Txy[iBlock-1][jBlock];
				}

				D_Txx	 = local_Txx[iBlock+1][jBlock]	- local_Txx[iBlock][jBlock];
				D_Tyy	 = local_Tyy[iBlock][jBlock+1]	- local_Tyy[iBlock][jBlock];

				tvx = vx[iField*NRI + jField];
			    tvy = vy[iField*NRI + jField];

				tvx += (BX[iField*NRI+jField]*(dtx))*( D_Txx	 +	Dy_Txy );
				tvy += (BY[iField*NRI+jField]*(dtx))*( Dx_Txy	 +	D_Tyy  );

			    
			   tabs = ABS[iField*NRI + jField];
			   vx[iField*NRI + jField]		= tvx*tabs;
			   vy[iField*NRI + jField]		= tvy*tabs;
    }

}

/** Stress Kernel (Local Memory) for Elastic Case
*/

__kernel void Stress_Local(__global float *Txx,  __global float *Txy,  __global float *Tyy,
						   __global float *vx,   __global float *vy,   
						   __global const float *C11, __global const float *C12, __global const float *C44,
						   __global const float *ABS){


		 __local  float local_vx[DimLocalX+Stencil][DimLocalY+Stencil];
		 __local  float local_vy[DimLocalX+Stencil][DimLocalY+Stencil];
		 __local  float local_ABS[DimLocalX+Stencil][DimLocalY+Stencil];
		

		 uint halo = Stencil/2;
		 uint j	   = get_group_id(0) * get_local_size(0) + get_local_id(0);
		 uint i	   = get_group_id(1) * get_local_size(1) + get_local_id(1);

		 uint iBlock			   = get_local_id(1)+halo;
		 uint jBlock			   = get_local_id(0)+halo;

		 uint iField			   = i + halo;
		 uint jField			   = j + halo;

		 float D_vx				  = 0.0f;
		 float D_vy				  = 0.0f;
		 float Dy_vx			  = 0.0f;
		 float Dx_vy			  = 0.0f;

		float tabs,tTxx, tTyy, tTxy;


		  if (i <  MRI - Stencil && j < NRI - Stencil ){


				barrier(CLK_LOCAL_MEM_FENCE);

				// fill left halo of local memory
				if (get_local_id(0) < halo )
				{
							  local_vx[iBlock][jBlock-halo]  = vx[iField*NRI + jField - halo];
							  local_vy[iBlock][jBlock-halo]  = vy[iField*NRI + jField - halo];
							  
							  
				}

				// fill right halo of local memory
				if (get_local_id(0) > DimLocalY - halo -1 )
				{
							  local_vx[iBlock][jBlock+halo]  = vx[iField*NRI + jField + halo];
							  local_vy[iBlock][jBlock+halo]  = vy[iField*NRI + jField + halo];
							 
				}

				// fill bottom halo of local memory
				if (get_local_id(1) < halo )
				{
							  local_vx[iBlock-halo][jBlock]  = vx[(iField-halo)*NRI + jField];
							  local_vy[iBlock-halo][jBlock]  = vy[(iField-halo)*NRI + jField];
						
				}


				// fill top halo of local memory
				if (get_local_id(1) > DimLocalX - halo -1 )
				{
							  local_vx[iBlock+halo][jBlock]  = vx[(iField+halo)*NRI + jField];
							  local_vy[iBlock+halo][jBlock]  = vy[(iField+halo)*NRI + jField];
							  
				}


				local_vx[iBlock][jBlock]  = vx[iField*NRI + jField];
				local_vy[iBlock][jBlock]  = vy[iField*NRI + jField];
				
				barrier(CLK_LOCAL_MEM_FENCE);

				D_vx   = 0.0f;
				D_vy   = 0.0f;
				Dy_vx  = 0.0f;
				Dx_vy  = 0.0f;

				if (i!= 0){
							 D_vx = local_vx[iBlock][jBlock]- local_vx[iBlock-1][jBlock];
				}

				if (j!=0){
							 D_vy = local_vy[iBlock][jBlock]- local_vy[iBlock][jBlock-1];
				}


				 Dy_vx = local_vx[iBlock][jBlock+1] - local_vx[iBlock][jBlock];
				 Dx_vy = local_vy[iBlock+1][jBlock] - local_vy[iBlock][jBlock];

				 tTxx = Txx[iField*NRI + jField];
				 tTyy = Tyy[iField*NRI + jField];
				 tTxy = Txy[iField*NRI + jField];

				 tTxx += (C11[iField*NRI + jField]*dtx)*D_vx + (C12[iField*NRI + jField]*dtx)*D_vy ;
				 tTyy += (C12[iField*NRI + jField]*dtx)*D_vx + (C11[iField*NRI + jField]*dtx)*D_vy ;
				 tTxy += (C44[iField*NRI + jField]*dtx)*( Dx_vy + Dy_vx );

				 tabs = ABS[iField*NRI + jField];

				 Txx[iField*NRI + jField]	= tTxx *  tabs;
				 Tyy[iField*NRI + jField]   = tTyy * tabs;
				 Txy[iField*NRI + jField]	= tTxy * tabs;
          }

}




		 
