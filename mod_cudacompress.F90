module cudacompress
   use cudafor
   implicit none

   private
   
   integer(kind=int32), parameter :: P = 17_int32
   integer(kind=int32), parameter :: T = 18_int32
   integer(kind=int32), parameter :: E =  7_int32
   integer(kind=int32), parameter :: M = 22_int32
   integer(kind=int32), parameter :: B = 80_int32
   integer(kind=int32), parameter :: TP = T + P
   integer(kind=int32), parameter :: TPM = T + P + M
   
   integer(kind=int32), parameter :: tm = ishft(1_int32,T) - 1_int32
   integer(kind=int32), parameter :: pm = ishft(1_int32,P) - 1_int32
   integer(kind=int32), parameter :: em = ishft(1_int32,E) - 1_int32
   integer(kind=int32), parameter :: mm = ishft(1_int32,M) - 1_int32

   integer(kind=int32), parameter :: ntmax = ishft(1_int32,T) - 1_int32
   integer(kind=int32), parameter :: npmax = ishft(1_int32,P) - 1_int32
   
   real(kind=real32), parameter :: ntmaxs = real(ntmax,kind=real32)
   real(kind=real32), parameter :: npmaxs = real(npmax,kind=real32)
   real(kind=real32), parameter :: trntmaxs = 2e0_real32/real(ntmax,kind=real32)
   real(kind=real32), parameter :: rnpmaxs = 1e0_real32/real(npmax,kind=real32)
   
   real(kind=real32), parameter :: pis = 4e0_real32*atan(1e0_real32)
   real(kind=real32), parameter :: rpis = 1e0_real32/pis
   real(kind=real32), parameter :: hrpis = 0.5e0_real32/pis

   real(kind=real32), parameter :: tol = 1e-12_real32
   
   public :: compress,decompress
  
contains
   !**********************************************************************
   attributes(device) function compress(x) result(c)
      use, intrinsic :: iso_fortran_env, only : real32,real64,int32,int64
      use cudadevice, only : __float_as_int
      implicit none
     
      real(kind=real32), intent(in) :: x(3)

      integer(kind=int64) :: c

      integer(kind=int32) :: ir
      integer(kind=int64) :: ex,mn,np,nt

      real(kind=real64) :: x2(3),r

      x2(1) = real(x(1),kind=real64)
      x2(2) = real(x(2),kind=real64)
      x2(3) = real(x(3),kind=real64)
      
      r = sqrt(x2(1)*x2(1) + x2(2)*x2(2) + x2(3)*x2(3))
      
      nt = nint(ntmaxs*(atan2(x(2),x(1))*hrpis + 0.5e0_real32))
      if(r .lt. tol) then
         np = nint(npmaxs*0.5_real32)
      else
         np = nint(npmaxs*rpis*real(acos(x2(3)/r),kind=real32))
      endif
         
      ir = __float_as_int(real(r,kind=real32))
      ex = int(ishft(iand(ir,z'7F800000'), -23),kind=int64) - 127 + B
      mn = int(ishft(iand(ir,z'007FFFFF'),M-23),kind=int64)

      c = ior(nt,ishft(np,T))
      c = ior(c,ishft(mn,TP))
      c = ior(c,ishft(ex,TPM))
      
      return
   end function compress
   !**********************************************************************
   attributes(device) function decompress(c) result(x)
      use, intrinsic :: iso_fortran_env, only : real32,real64,int32,int64
      use cudadevice, only : __int_as_float
      implicit none

      integer(kind=int64), intent(in) :: c
      
      real(kind=real32) :: x(3)

      integer(kind=int32) :: ir
      integer(kind=int32) :: ex,mn,np,nt
      
      real(kind=real32) :: r,phi,the
            
      nt = iand(int(c,kind=int32),tm)
      np = iand(int(ishft(c,-T),kind=int32),pm)
      the = pis*(real(nt,kind=real32)*trntmaxs - 1e0_real32)
      phi = pis*real(np,kind=real32)*rnpmaxs
      
      ex = iand(int(ishft(c,-TPM),kind=int32),em)
      ex = ishft(ex - B + 127,23)
      mn = iand(int(ishft(c,-TP),kind=int32),mm)
      mn = ishft(mn,23-M)
      r = __int_as_float(ior(ex,mn))

      x(1) = r*cos(the)*sin(phi)
      x(2) = r*sin(the)*sin(phi)
      x(3) = r*cos(phi)
      
      return
   end function decompress
   !**********************************************************************
end module cudacompress
