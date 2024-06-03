import tqdm
import numpy as np
import xarray as xr
import pandas as pd
import tqdm
import pandas as pd
import DeepDyamond
from scipy import signal
import math

from scipy.interpolate import interp2d

def get_area_mask_array(mask,threshold2=0.4):
    idx_contour = np.where(mask>threshold2)
    mask_binary = np.zeros_like(mask)
    mask_binary[idx_contour] = 1
    return mask_binary


def get_perimeter_mask_array(mask, threshold1=0.6,threshold2=0.4):
    idx_contour = np.where((mask<threshold1) & (mask>threshold2))
    mask_binary = np.zeros_like(mask)
    mask_binary[idx_contour] = 1
    return mask_binary


def get_area_mask(mask):
    binary_mask = get_area_mask_array(mask)
    return np.sum(binary_mask)

def get_perimeter_mask(mask):
    perimeter_array = get_perimeter_mask_array(mask)
    return np.sum(perimeter_array)

def get_radius_eq(mask):
    area = get_area_mask(mask)
    R=(area/np.pi)**0.5
    return R


def get_circularity_parameter_mask(mask):
    area = get_area_mask(mask)
    perimeter = get_perimeter_mask(mask)
    
    p = 4*np.pi*area / (perimeter)**2
    
    if p>1:
        return np.nan
    
    return p 



def get_changing_shape_same_area(A0, XX_1):
    
    X_1 = get_area_mask_array(XX_1)

    nx, ny = X_1.shape
    X_grid=np.linspace(-0.5,0.5,nx)
    Y_grid=np.linspace(-0.5,0.5,ny)


    A1 = get_area_mask(X_1)
    lambda_factor = (A0/A1)**0.5

    X1_streched = interp2d(X_grid*lambda_factor,Y_grid*lambda_factor, X_1)
    X1_tilda = X1_streched(X_grid,Y_grid)


    return get_area_mask_array(X1_tilda)


def get_same_inital_shape_changing_area(XX_0, A1):
    
    X_0 = get_area_mask_array(XX_0)
    
    n_x, n_y = X_0.shape
    
    X_grid=np.linspace(-0.5,0.5,n_x)
    Y_grid=np.linspace(-0.5,0.5,n_y)

    A0 = get_area_mask(X_0)
    
    lambda_factor = (A0/A1)**0.5
    
    X_streched = interp2d(X_grid*lambda_factor,Y_grid*lambda_factor, X_0)
    X_tilda = X_streched(X_grid,Y_grid)
    
    return get_area_mask(X_tilda)

def get_circle_shape_changing_area(XX_0):
    nx, ny = XX_0.shape
    px,py = int(nx/2), int(ny/2)
    
    def cercle(R,p=(px,py), nx=nx,ny=ny):
        def dist(p,q):
            return math.dist(p, q)
        M=np.zeros((nx,ny))
        for i in range(nx):
            for j in range(ny):
                if dist(p, (i,j))<R:
                    M[i,j]=1
                else:
                    None            
        return M

    #X_0 = get_area_mask_array(XX_0)

    R_eq_0 = get_radius_eq(XX_0)      
    
    C_0 = cercle(R_eq_0)

    return(C_0)