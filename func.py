import numpy as np
import math
from matplotlib import pyplot as plt
import meep as mp
from meep import mpb
import time
import pandas as pd

def lateral_shift(offset1,offset2,nrows=2,resolution=10,k_resolution=20,num_bands=42,width=7,debug=False,radius=0.34,dir='old',points='full',twod=False):
    
    '''
    function to return bands for a given lateral offset
    
    '''
    
    
    start_time=time.time()
    bands_list=[]
    r = radius
            
    a=422
    h_dl = 220/a
        
        # Materials
    Si = mp.Medium(index=3.48) #3.48
    Gamma = mp.Vector3(0, 0)
    M = mp.Vector3(1/2, -1/2)
    
    
        
    # Triangular lattice:
    lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                      basis1=mp.Vector3(math.sqrt(3)/2, 1/2),
                                      basis2=mp.Vector3(math.sqrt(3)/2, -1/2))
        
        # Transform the k-points defined above into the reciprocal space:
    #Gamma = mp.cartesian_to_reciprocal(Gamma, lattice)
    if dir=='old':
        
        K = mp.Vector3(1/2, 0)
        K = mp.cartesian_to_reciprocal(K, lattice)
        Gamma=mp.cartesian_to_reciprocal(K,lattice)
        # Define the path in k-space:
        # k_points
    else:
        K=mp.Vector3(-1.0/3,1.0/3)   
    
    
    if points=='full':
        
        k_points = [Gamma,   K,       # K
                    #mp.Vector3(0, 0.5),  # K
                    #mp.Vector3(-0.5)           # Gamma
                       ]
    else:
        k_points = [K/2,          # Gamma
                        K,       # K
                    #mp.Vector3(0, 0.5),  # K
                    #mp.Vector3(-0.5)           # Gamma
                       ]    
            # Interpolate points in between the high-symmetry points:
    k_points = mp.interpolate(k_resolution, k_points)
            # Simulation cell
         
    if nrows==2 or nrows==1:
        if twod:
            geometry_lattice=mp.Lattice(size=mp.Vector3(2*(np.sqrt(3))*(width-0.5), 1))
        else:
            geometry_lattice=mp.Lattice(size=mp.Vector3(2*(np.sqrt(3))*(width-0.5), 1,2))   
    else:
        if twod:
            geometry_lattice = mp.Lattice(size=mp.Vector3(2*(np.sqrt(3))*(width-0.5)-2*offset1, 1))   
        else:
            
            geometry_lattice = mp.Lattice(size=mp.Vector3(2*(np.sqrt(3))*(width-0.5)-2*offset1, 1,2))
            
            # Geometry
    l1 = mp.Cylinder(center=(-np.sqrt(3)/2, 0), radius=r, height=mp.inf, material=mp.air)
    l2 = mp.Cylinder(center=(-np.sqrt(3), -1/2), radius=r, height=mp.inf, material=mp.air)
    l3 = mp.Cylinder(center=(-np.sqrt(3), 1/2), radius=r, height=mp.inf, material=mp.air)
            
    r1 = mp.Cylinder(center=(np.sqrt(3)/2, 0), radius=r, height=mp.inf, material=mp.air)
    r2 = mp.Cylinder(center=(np.sqrt(3), -1/2), radius=r, height=mp.inf, material=mp.air)
    r3 = mp.Cylinder(center=(np.sqrt(3), 1/2), radius=r, height=mp.inf, material=mp.air)
            
    c1 = mp.Cylinder(center=(0, -1/2), radius=r, height=mp.inf, material=mp.air)
    c2 = mp.Cylinder(center=(0, 1/2), radius=r, height=mp.inf, material=mp.air)
            
    l01=mp.Cylinder(center=(-np.sqrt(3)/2+offset1, 0), radius=r, height=mp.inf, material=mp.air)
    r01=mp.Cylinder(center=(np.sqrt(3)/2-offset1, 0), radius=r, height=mp.inf, material=mp.air)
            
    l02 = mp.Cylinder(center=(-np.sqrt(3)+offset2, -1/2), radius=r, height=mp.inf, material=mp.air)
    l03 = mp.Cylinder(center=(-np.sqrt(3)+offset2, 1/2), radius=r, height=mp.inf, material=mp.air)
            
    r02 = mp.Cylinder(center=(np.sqrt(3)-offset2, -1/2), radius=r, height=mp.inf, material=mp.air)
    r03 = mp.Cylinder(center=(np.sqrt(3)-offset2, 1/2), radius=r, height=mp.inf, material=mp.air)
            
    if twod:        
        dl = mp.Block(size=(mp.inf, mp.inf), center=(0, 0), material=Si, e1=(1, 0), e2=(0, 1))   
    else:
        dl = mp.Block(size=(mp.inf, mp.inf, h_dl), center=(0, 0, 0), material=Si, e1=(1, 0, 0), e2=(0, 1, 0), e3=(0, 0, 1))
            
    geometry = [dl, l01, l02, l03, r01, r02, r03]#, c1, c2]
  
    if nrows==1:
        geometry = [dl, l01, l2, l3, r01, r2, r3]#, c1, c2]    
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(-np.sqrt(3), 0), 1, width, [l1, l2, l3]))
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(np.sqrt(3), 0), 1, width, [r1, r2, r3]))
    if nrows==2:
        geometry = [dl, l01, l02, l03, r01, r02, r03]#, c1, c2] 
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(-np.sqrt(3), 0), 1, width, [l1, l2, l3]))
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(np.sqrt(3), 0), 1, width, [r1, r2, r3]))

    else:
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(-np.sqrt(3), 0), 1, width, [l01, l02, l03]))
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(np.sqrt(3), 0), 1, width, [r01, r02, r03]))

            # Mode solver
            
    if debug:
        s = mp.Source(src=mp.GaussianSource(1, fwidth=1), component=mp.Hz, center=mp.Vector3(0.1234, 0.2345))
        if nrows==2 or nrows==1:
            sim = mp.Simulation(cell_size=mp.Vector3(2*(np.sqrt(3))*(width-0.5), 1), geometry=geometry, sources=[], symmetries=[],
                        boundary_layers=[], resolution=resolution, default_material=Si)
            sim.plot2D()
    
            plt.show()       
        else:
            sim = mp.Simulation(cell_size=mp.Vector3(2*(np.sqrt(3))*(width-0.5)-2*offset1, 1), geometry=geometry, sources=[], symmetries=[],
                        boundary_layers=[], resolution=resolution, default_material=Si)
            sim.plot2D()
    
            plt.show()   
    else:        
        ms = mpb.ModeSolver(num_bands=num_bands,
                                k_points=k_points,
                                geometry=geometry,
                                geometry_lattice=geometry_lattice,
                                resolution=resolution,
                                tolerance=1.0e-5,
                                default_material=mp.air)
        ms.run_te()
        print('\n\n')
        print('\n Bands calculated for grid point ', offset1, offset2 )
        
        
        end_time=time.time()

        print('Time taken (min) ', (end_time-start_time)/60)
        
        
        bands=np.transpose(ms.all_freqs)
        
        return bands

def longitudnal_shift(offset1,offset2,nrows=2,resolution=10,k_resolution=20,num_bands=42,width=7,debug=False,radius=0.34,dir='old',twod=False):
    
    bands_list=[]
    
    
    
    
    mp.verbosity(0)
        
    r = radius
    a=422        
    h_dl = 220/a
        
        # Materials
    Si = mp.Medium(index=3.48) #3.48
    Gamma = mp.Vector3(0, 0)
    M = mp.Vector3(1/2, -1/2)
   
        
    # Triangular lattice:
    lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                      basis1=mp.Vector3(math.sqrt(3)/2, 1/2),
                                      basis2=mp.Vector3(math.sqrt(3)/2, -1/2))
        
        # Transform the k-points defined above into the reciprocal space:
    
    if dir=='old':
        
        K = mp.Vector3(1/2, 0)
        K = mp.cartesian_to_reciprocal(K, lattice)
        Gamma=mp.cartesian_to_reciprocal(K,lattice)
    else:
        K=mp.Vector3(-1.0/3,1.0/3)
    #K = mp.cartesian_to_reciprocal(K, lattice)
        
        # Define the path in k-space:
        # k_points
    k_points = [Gamma,          # Gamma
                    K,       # K
                    #mp.Vector3(0, 0.5),  # K
                    #mp.Vector3(-0.5)           # Gamma
                   ]
        
            # Interpolate points in between the high-symmetry points:
    k_points = mp.interpolate(k_resolution, k_points)
            # Simulation cell
         
    if twod:
        geometry_lattice = mp.Lattice(size=mp.Vector3(2*np.sqrt(3)*(width-0.5), 1))    
    else:
        geometry_lattice = mp.Lattice(size=mp.Vector3(2*np.sqrt(3)*(width-0.5), 1,2))
            
            # Geometry
    l1 = mp.Cylinder(center=(-np.sqrt(3)/2, 0), radius=r, height=mp.inf, material=mp.air)
    l2 = mp.Cylinder(center=(-np.sqrt(3), -1/2), radius=r, height=mp.inf, material=mp.air)
    l3 = mp.Cylinder(center=(-np.sqrt(3), 1/2), radius=r, height=mp.inf, material=mp.air)
            
    r1 = mp.Cylinder(center=(np.sqrt(3)/2, 0), radius=r, height=mp.inf, material=mp.air)
    r2 = mp.Cylinder(center=(np.sqrt(3), -1/2), radius=r, height=mp.inf, material=mp.air)
    r3 = mp.Cylinder(center=(np.sqrt(3), 1/2), radius=r, height=mp.inf, material=mp.air)
            
    c1 = mp.Cylinder(center=(0, -1/2), radius=r, height=mp.inf, material=mp.air)
    c2 = mp.Cylinder(center=(0, 1/2), radius=r, height=mp.inf, material=mp.air)
            
    l01=mp.Cylinder(center=(-np.sqrt(3)/2, offset1), radius=r, height=mp.inf, material=mp.air)
    r01=mp.Cylinder(center=(np.sqrt(3)/2, offset1), radius=r, height=mp.inf, material=mp.air)
            
    l02 = mp.Cylinder(center=(-np.sqrt(3), -1/2+offset2), radius=r, height=mp.inf, material=mp.air)
    l03 = mp.Cylinder(center=(-np.sqrt(3), 1/2+offset2), radius=r, height=mp.inf, material=mp.air)
            
    r02 = mp.Cylinder(center=(np.sqrt(3), -1/2+offset2), radius=r, height=mp.inf, material=mp.air)
    r03 = mp.Cylinder(center=(np.sqrt(3), 1/2+offset2), radius=r, height=mp.inf, material=mp.air)
            
    if twod:        
        dl = mp.Block(size=(mp.inf, mp.inf), center=(0, 0), material=Si, e1=(1, 0), e2=(0, 1))  
    else:
        dl = mp.Block(size=(mp.inf, mp.inf, h_dl), center=(0, 0, 0), material=Si, e1=(1, 0, 0), e2=(0, 1, 0), e3=(0, 0, 1))
            
    geometry = [dl, l01, l02, l03, r01, r02, r03]#, c1, c2]
    if nrows==1:
        geometry = [dl, l01, l2, l3, r01, r2, r3]#, c1, c2]    
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(-np.sqrt(3), 0), 1, width, [l1, l2, l3]))
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(np.sqrt(3), 0), 1, width, [r1, r2, r3]))
    if nrows==2:
        geometry = [dl, l01, l02, l03, r01, r02, r03]#, c1, c2] 
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(-np.sqrt(3), 0), 1, width, [l1, l2, l3]))
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(np.sqrt(3), 0), 1, width, [r1, r2, r3]))

    else:
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(-np.sqrt(3), 0), 1, width, [l01, l02, l03]))
        geometry.extend(mp.geometric_objects_duplicates(mp.Vector3(np.sqrt(3), 0), 1, width, [r01, r02, r03]))

            # Mode solver
            
    if debug:
        s = mp.Source(src=mp.GaussianSource(1, fwidth=1), component=mp.Hz, center=mp.Vector3(0.1234, 0.2345))
        sim = mp.Simulation(cell_size=mp.Vector3(2*np.sqrt(3)*(width-0.5), 1), geometry=geometry, sources=[], symmetries=[],
                    boundary_layers=[], resolution=resolution, default_material=Si)
        sim.plot2D()

        plt.show()   
    else:        
        ms = mpb.ModeSolver(num_bands=num_bands,
                                k_points=k_points,
                                geometry=geometry,
                                geometry_lattice=geometry_lattice,
                                resolution=resolution,
                                tolerance=1.0e-5,
                                default_material=mp.air)
        ms.run_te()
        print('\n\n')
        print('\n Bands calculated for grid point ', offset1, offset2 )
        
           #plt.savefig('20200404_Slice_W1_MPB_3D_r_0-25_te_nSi-3-48_w8_res16_h2_zodd.png', dpi=100, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        
        
        
        
        
        
        
        bands=np.transpose(ms.all_freqs)
            
        return bands

def run_lateral_sweep(offset_1_lim=1.73*0.1,offset_2_lim=1.73*0.1,n1=5,n2=5,resolution=10,k_resolution=28,num_bands=50,width=11,radius=0.34,
                  nrows=2):
    offset1_grid=np.linspace(0.0,offset_1_lim,n1)
    offset2_grid=np.linspace(0.0,offset_2_lim,n2)
    band_list=[]
    for offset1 in offset1_grid:
        for offset2 in offset2_grid:
            bands=lateral_shift(offset1,offset2,nrows,resolution,k_resolution,num_bands,width,radius=radius)
            band_list.append(bands)

    band_list=np.array(band_list)
    
    full_bands=np.reshape(band_list,(n1,n2,band_list[0].shape[0],band_list[0].shape[1]))
    np.savez(f'latsweep_{offset_1_lim}_{offset_2_lim}_{n1}_{n2}_{width}_res={resolution}_rad={radius}', full_bands)
    return full_bands


def run_longitudnal_sweep(offset_1_lim=0.6,offset_2_lim=0.6,n1=5,n2=5,resolution=10,k_resolution=28,num_bands=42,width=7,radius=0.34,nrows=2):
    offset1_grid=np.linspace(0.0,offset_1_lim,n1)
    offset2_grid=np.linspace(0.0,offset_2_lim,n2)
    band_list=[]
    for offset1 in offset1_grid:
        for offset2 in offset2_grid:
            bands=longitudnal_shift(offset1,offset2,nrows,resolution,k_resolution,num_bands,width,radius=radius)
            band_list.append(bands)

    band_list=np.array(band_list)
    
    full_bands=np.reshape(band_list,(n1,n2,band_list[0].shape[0],band_list[0].shape[1]))
    np.savez(f'longsweep_{offset_1_lim}_{offset_2_lim}_{n1}_{n2}_{width}_res={resolution}_rad={radius}', full_bands)
    return full_bands





def plot_band_structure(bands,band_index,freq='frequency',velocity=False,zoom=False,C_band='Full',dir='old',test=False,plot=True, abs=True):
    
    k_resolution = len(bands[0])-2
    #bg1_min = ms.gap_list[0][1]
    #bg1_max = ms.gap_list[0][2]
    #bg1_center = (bg1_min + bg1_max)/2
    
    a=422
    wavelength=1510
    wavelength_cutoff=1400
    plt.figure(figsize=(8,6))
    M = mp.Vector3(1/2, -1/2)
    lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                      basis1=mp.Vector3(math.sqrt(3)/2, 1/2),
                                      basis2=mp.Vector3(math.sqrt(3)/2, -1/2))
    Gamma=mp.Vector3()  
    if dir=='old':
        K=mp.Vector3(1/2,0)
        K = mp.cartesian_to_reciprocal(K, lattice)
        Gamma = mp.cartesian_to_reciprocal(Gamma, lattice)
    else:
    
        K = mp.Vector3(-1.0/3, 1.0/3)
    
    Gamma=mp.Vector3()   
        # Triangular lattice:
   
        # Transform the k-points defined above into the reciprocal space:
    #Gamma = mp.cartesian_to_reciprocal(Gamma, lattice)
    #K = mp.cartesian_to_reciprocal(K, lattice)
    
        # Define the path in k-space:
        # k_points
    k_points = [Gamma,          # Gamma
                    K,       # K
                    #mp.Vector3(0, 0.5),  # K
                    #mp.Vector3(-0.5)           # Gamma
                   ]
        
            # Interpolate points in between the high-symmetry points:
    k_points = mp.interpolate(k_resolution, k_points)
    nk = len(k_points)
    
  
        
    k_abs=[np.linalg.norm(mp.reciprocal_to_cartesian(k,lattice)) for k in k_points]
    
    if plot==True:
        for band in bands:
            #plt.plot(np.arange(1,nk+1), band, color='blue')
            plt.plot(k_abs,band,color='blue')   
        
        plt.plot(k_abs, bands[band_index], color='orange')
    
        plt.plot(k_abs[:-5],k_abs[:-5],color='red')
        
    #plt.axvline(x=k_resolution+2, ymin=0, ymax=1, color='black', linestyle=':', linewidth=1)
    #plt.axvline(x=2*k_resolution+3, ymin=0, ymax=1, color='black', linestyle=':', linewidth=1)
    
    #plt.axhspan(ymin=bg1_min, ymax=bg1_max, xmin=0, xmax=16, color='yellow', alpha=0.5)
    #plt.axhline(y=bg1_max, xmin=0, xmax=16, color='black', linestyle='--', linewidth=1)
    #plt.text(7.5, bg1_center-0.007, 'Band gap', size=12)
        
        plt.xlabel('Bloch wavevector')
        #plt.xticks([1,k_resolution+2,2*k_resolution+3,3*k_resolution+4], [r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
        #plt.xticks([1,30], [r'$\Gamma$', 'M'])
        plt.ylabel(r'Frequency $[a/\lambda]$')
        
        #plt.ylim(0.25, 0.35)
        
        #plt.savefig('20200404_Slice_W1_MPB_3D_r_0-25_te_nSi-3-48_w8_res16_h2_zodd.png', dpi=100, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        
        plt.show()

    
    k_values = k_points
    omega_values=np.array(bands[band_index])
    k_abs=np.array(k_abs)
    # Calculate the group velocity for the chosen mode
    # Using finite differences: v_g = (omega[i+1] - omega[i]) / (k[i+1] - k[i])
 
    
    if zoom==True:
        light_line_index=[i for i,omega in enumerate(omega_values) if omega<k_abs[i]]
        # Display the group velocity for the chosen mode
        
        omega_values=omega_values[light_line_index]
        k_abs=k_abs[light_line_index]
        if plot==True:
            for band in bands:
            #plt.plot(np.arange(1,nk+1), band, color='blue')
                plt.plot(k_abs,band[light_line_index],color='blue')   
        
            plt.plot(k_abs, bands[band_index][light_line_index], color='orange')
            plt.show()
    group_velocity=[]
    for i in range(len(k_abs)-1):
        group_velocity.append(-(omega_values[i+1] - omega_values[i]) / (k_abs[i+1] - k_abs[i]))
    
    if abs==True:
        group_velocity=np.abs(np.array(group_velocity))    
    else:
        group_velocity=np.array(group_velocity)
    
    
        

    
    

    variation_threshold = 0.1  # 10% variation
    
    group_index=1/group_velocity
    center=0
    left_freq=0
    right_freq=0
    left_index=0
    right_index=0
    l=-1
    max_group_index=1
    
    
    
    
    if C_band=='Full':
    
        for i,vel in enumerate(group_velocity):
            
            index=1/vel
            print(i, 'group_index =',1/vel ,'v=', vel)
            if i>0 and i<len(group_velocity):
                
                max_magnitude= np.abs(vel)/(1-variation_threshold)
                
                min_magnitude =np.abs (vel)/(1+variation_threshold)
        
                
                # Determine the indices for the specified frequency range
                
                left=i
                right=i
                while(np.abs(group_velocity[left])< max_magnitude and np.abs(group_velocity[left]) > min_magnitude):
                    if left>0:
                        left=left-1
                                    
                    else:
                        break
                while(np.abs(group_velocity[right])<max_magnitude and np.abs(group_velocity[right])>min_magnitude):
                    if right<len(group_velocity)-1:
                        right=right+1
                    else:
                        break
                
                if np.abs(group_velocity[left])< max_magnitude and np.abs(group_velocity[left]) > min_magnitude and left==0:
                    pass
                else:
                    left=left+1
                
                if (right==len(group_index)-1 and np.abs(group_velocity[right])<max_magnitude and np.abs(group_velocity[right])>min_magnitude):
                    print('flag')
                    
                else:
                    right=right-1
                
                
                
                print('left=',left,'center=',i,'right=',right )
                bandwidth=omega_values[left+1]-omega_values[right+1]
               
                product=np.abs(index*bandwidth)/omega_values[i+1]
                
                if product>l:
                    max_group_index=index
                    l=product
                    left_freq=omega_values[left+1]
                    right_freq=omega_values[right+1]
                    left_index=left
                    right_index=right
                    center=i

    # Calculate bandwidth (BW) and center frequency (Fc)
    elif C_band=='near':
        C_band_wavelengths=a/omega_values[a/omega_values>wavelength_cutoff] 
        i=np.where(a/omega_values==C_band_wavelengths[0])[-1][0]-1
        
        for i in range(i,len(group_index)):
            index=group_index[i]
            print(i, 'group_index =',index)
            if i>0 and i<len(group_index):
                
                max_magnitude= np.abs(index)+ (np.abs(index)*variation_threshold)
                min_magnitude = np.abs(index) - np.abs((index* variation_threshold))
        
                
                # Determine the indices for the specified frequency range
                
                left=i
                right=i
                while(np.abs(group_index[left])< max_magnitude and np.abs(group_index[left]) > min_magnitude):
                    if left>0:
                        left=left-1
                                    
                    else:
                        break
                while(np.abs(group_index[right])<max_magnitude and np.abs(group_index[right])>min_magnitude):
                    if right<len(group_index)-1:
                        right=right+1
                    else:
                        break
                
                left=left+1
                right=right-1
                
                
                print('left=',left,'center=',i,'right=',right )
                bandwidth=omega_values[left+1]-omega_values[right+1]
               
                product=np.abs(index*bandwidth)/omega_values[i+1]
                
                if product>l:
                    max_group_index=index
                    l=product
                    left_freq=omega_values[left+1]
                    right_freq=omega_values[right+1]
                    left_index=left
                    right_index=right
                    center=i

    
    elif C_band=='fixed':
        for delta in np.linspace(0,10,50):
            
           
            C_band_wavelengths=a/omega_values[a/omega_values<wavelength+delta] 
            C_band_wavelengths=C_band_wavelengths[C_band_wavelengths>wavelength-delta]   
            
            if len(C_band_wavelengths)>=1:
                
                i=np.where(a/omega_values==C_band_wavelengths[0])[-1][0]-1
                index=group_index[i]
                max_magnitude= index+ (index*variation_threshold)
                min_magnitude = index - (index* variation_threshold)
                print(i)
                    
                    # Determine the indices for the specified frequency range
                    
                left=i
                right=i
                while(group_index[left]<= max_magnitude and group_index[left] >= min_magnitude):
                    if left>0:
                        left=left-1
                                    
                    else:
                        break
                while(group_index[right]<=max_magnitude and group_index[right]>=min_magnitude):
                    if right<len(group_index)-1:
                        right=right+1
                    else:
                        break
                
                left=left+1
                if (right==len(group_index)-1 and group_index[right]<=max_magnitude and group_index[right]>=min_magnitude):
                    pass
                    
                else:
                    right=right-1
                
                print('left=',left,'center=',i,'right=',right )
                bandwidth=omega_values[left+1]-omega_values[right+1]
               
                product=np.abs(index*bandwidth)/omega_values[i+1]
            
                max_group_index=index
                l=product
                left_freq=omega_values[left+1]
                right_freq=omega_values[right+1]
                left_index=left
                right_index=right
                center=i
            
                break
    
    print('Bandwidth Product:', l)
    print('Bandwidth (BW):', l/(max_group_index*omega_values[center+1]))
    print('group_index',max_group_index)
    print('bandwidth um ' ,-1/omega_values[left_index+1]+1/omega_values[right_index+1])           
    print('frequency=', omega_values[center+1])
    if test==True:
        
        l=-1
        
        group_velocity=np.insert(group_velocity,0,group_velocity[0])
        df=pd.DataFrame({'k':k_abs,'w':omega_values,'v':group_velocity})
        df=df.sort_values(by='w')   
        df=df.reset_index(drop=True)
        if C_band=='Full':
    
            for i,omega in enumerate(df['w']):
                
                
                vel=df['v'].loc[i]
                print(vel)
                index=1/vel
                print(i, 'group_index =',1/vel ,'v=', vel)
                
                if i>0 and i<len(df['w']):
                    
                    max_magnitude= np.abs(vel)/(1-variation_threshold)
                    
                    min_magnitude =np.abs (vel)/(1+variation_threshold)
            
                    
                    # Determine the indices for the specified frequency range
                    
                    left=i
                    right=i
                    while(np.abs(df['v'].loc[left])< max_magnitude and np.abs(df['v'].loc[left]) > min_magnitude):
                        if left>0:
                            left=left-1
                                        
                        else:
                            break
                    while(np.abs(df['v'].loc[right])<max_magnitude and np.abs(df['v'].loc[right])>min_magnitude):
                        if right<len(df['w'])-1:
                            right=right+1
                        else:
                            break
                    
                    left=left+1
                    if (np.abs(df['v'].loc[right])<max_magnitude and np.abs(df['v'].loc[right])>min_magnitude and right==len(df['w'])-1):
                        pass
                    else:
                        
                    
                        right_=right-1
                    
                    
                    print('left=',left,'center=',i,'right=',right )
                    bandwidth=df['w'].loc[left]-df['w'].loc[right]
                   
                    product=np.abs(index*bandwidth)/omega
                    
                    if product>l:
                        max_group_index=index
                        l=product
                        left_freq=df['w'][left]
                        right_freq=df['w'][right]
                        left_index=left
                        right_index=right
                        center=i
        
            
        print('Bandwidth Product:', l)
        print('Bandwidth (BW):', l/(max_group_index*df['w'][center]))
        print('group_index',max_group_index)
        print('bandwidth nm ' ,-422/left_freq+422/right_freq)

        
        
        if plot==True:
            if freq=='frequency':
                if velocity:
                    plt.plot(df['w'],df['v'])
                    plt.xlabel('frequency')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=df['w'].loc[center],linestyle='--')
                else:
                    plt.plot(df['w'],1/df['v'])
                    plt.xlabel('frequency')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=df['w'].loc[center],linestyle='--')
            
            elif freq=='wave':
            
                if velocity:
                    plt.plot(a/df['w'],df['v'])
                    plt.xlabel('wavelength')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=a/left_freq,c='g',label='bandwidth')
                    plt.axvline(x=a/right_freq,c='g')
                    plt.axvline(x=a/df['w'].loc[center],linestyle='--')
                else:
                    plt.plot(a/df['w'],1/df['v'])
                    plt.xlabel('wavelength')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=a/left_freq,c='g',label='bandwidth')
                    plt.axvline(x=right_freq,c='g')
                    plt.axvline(x=a/df['w'].loc[center],linestyle='--')
                    
            else:
            
                if velocity==False:
                    plt.scatter(df['k'],1/df['v'])
                    plt.xlabel('k')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=df['k'].loc[left_index])
                    plt.axvline(x=df['k'].loc[right_index])
                    plt.axvline(x=df['k'].loc[center],linestyle='--')
                
                else:
                    
                    plt.scatter(df['k'] ,df['v'])
                    plt.xlabel('k')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=df['k'].loc[left_index])
                    plt.axvline(x=df['k'].loc[right_index])
                    plt.axvline(x=df['k'].loc[center],linestyle='--')
            
            plt.legend()    
        
    
    else:
        if plot==True:
            if freq=='frequency':
                if velocity:
                    plt.plot(omega_values[1:],group_velocity)
                    plt.xlabel('frequency')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=omega_values[center+1],linestyle='--')
                else:
                    plt.plot(omega_values[1:],1/group_velocity)
                    plt.xlabel('frequency')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=omega_values[center+1],linestyle='--')
            
            elif freq=='wave':
                
                if velocity:
                    plt.plot(422/omega_values[1:],group_velocity)
                    plt.xlabel('wavelength')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=a/omega_values[left_index+1],c='g',label='bandwidth')
                    plt.axvline(x=a/omega_values[right_index+1],c='g')
                    plt.axvline(x=a/omega_values[center+1],linestyle='--')
                else:
                    plt.plot(a/omega_values[1:],1/group_velocity)
                    plt.xlabel('wavelength')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=a/omega_values[left_index+1],c='g',label='bandwidth')
                    plt.axvline(x=a/omega_values[right_index+1],c='g')
                    plt.axvline(x=a/omega_values[center+1],linestyle='--')
                    
            else:
                
                if velocity==False:
                    plt.scatter(k_abs[1:],group_index)
                    plt.xlabel('k')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=k_abs[left_index+1])
                    plt.axvline(x=k_abs[right_index+1])
                    plt.axvline(x=k_abs[center+1],linestyle='--')
                
                else:
                    
                    plt.scatter(k_abs[1:],group_velocity)
                    plt.xlabel('k')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=k_abs[left_index+1])
                    plt.axvline(x=k_abs[right_index+1])
                    plt.axvline(x=k_abs[center+1],linestyle='--')
            plt.legend()
        
    return l,omega_values[center+1],max_group_index



def plot_group_index(band,freq='frequency',velocity=False,zoom=False,C_band='Full',dir='old',test=False,plot=True, cut=0,a=0.442):
    
    k_resolution = len(band)-2
    #bg1_min = ms.gap_list[0][1]
    #bg1_max = ms.gap_list[0][2]
    #bg1_center = (bg1_min + bg1_max)/2
    
    
    wavelength=1510
    wavelength_cutoff=1400
    plt.figure(figsize=(8,6))
    M = mp.Vector3(1/2, -1/2)
    lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                      basis1=mp.Vector3(math.sqrt(3)/2, 1/2),
                                      basis2=mp.Vector3(math.sqrt(3)/2, -1/2))
    Gamma=mp.Vector3()  
    if dir=='old':
        K=mp.Vector3(1/2,0)
        K = mp.cartesian_to_reciprocal(K, lattice)
        Gamma = mp.cartesian_to_reciprocal(Gamma, lattice)
    else:
    
        K = mp.Vector3(-1.0/3, 1.0/3)
    
    Gamma=mp.Vector3()   
        # Triangular lattice:
   
        # Transform the k-points defined above into the reciprocal space:
    #Gamma = mp.cartesian_to_reciprocal(Gamma, lattice)
    #K = mp.cartesian_to_reciprocal(K, lattice)
    
        # Define the path in k-space:
        # k_points
    k_points = [Gamma,          # Gamma
                    K,       # K
                    #mp.Vector3(0, 0.5),  # K
                    #mp.Vector3(-0.5)           # Gamma
                   ]
        
            # Interpolate points in between the high-symmetry points:
    k_points = mp.interpolate(k_resolution, k_points)
    nk = len(k_points)
    
  
        
    k_abs=[np.linalg.norm(mp.reciprocal_to_cartesian(k,lattice)) for k in k_points]
    
    
    
    k_values = k_points
    omega_values=np.array(band)
    k_abs=np.array(k_abs)
    # Calculate the group velocity for the chosen mode
    # Using finite differences: v_g = (omega[i+1] - omega[i]) / (k[i+1] - k[i])
 
    
    if zoom==True:
        light_line_index=[i for i,omega in enumerate(omega_values) if omega<k_abs[i]]
        # Display the group velocity for the chosen mode
        light_line_index=[i for i in light_line_index if k_abs[i]>cut]
        omega_values=omega_values[light_line_index]
        k_abs=k_abs[light_line_index]
        
        
    group_velocity=[]
    for i in range(len(k_abs)-1):
        group_velocity.append(-(omega_values[i+1] - omega_values[i]) / (k_abs[i+1] - k_abs[i]))
    
    
    group_velocity=np.abs(np.array(group_velocity))
    
    
        

    
    

    variation_threshold = 0.1  # 10% variation
    
    group_index=1/group_velocity
    center=0
    left_freq=0
    right_freq=0
    left_index=0
    right_index=0
    l=-1
    max_group_index=1
    
    
    
    
    if C_band=='Full':
    
        for i,vel in enumerate(group_velocity):
            
            index=1/vel
            
            if i>0 and i<len(group_velocity):
                
                max_magnitude= np.abs(vel)/(1-variation_threshold)
                
                min_magnitude =np.abs (vel)/(1+variation_threshold)
        
                
                # Determine the indices for the specified frequency range
                
                left=i
                right=i
                while(np.abs(group_velocity[left])< max_magnitude and np.abs(group_velocity[left]) > min_magnitude):
                    if left>0:
                        left=left-1
                                    
                    else:
                        break
                while(np.abs(group_velocity[right])<max_magnitude and np.abs(group_velocity[right])>min_magnitude):
                    if right<len(group_velocity)-1:
                        right=right+1
                    else:
                        break
                
                if np.abs(group_velocity[left])< max_magnitude and np.abs(group_velocity[left]) > min_magnitude and left==0:
                    pass
                else:
                    left=left+1
                
                if (right==len(group_index)-1 and np.abs(group_velocity[right])<max_magnitude and np.abs(group_velocity[right])>min_magnitude):
                    print('flag')
                    
                else:
                    right=right-1
                
                
                
                
                bandwidth=omega_values[left+1]-omega_values[right+1]
               
                product=np.abs(index*bandwidth)/omega_values[i+1]
                
                if product>l:
                    max_group_index=index
                    l=product
                    left_freq=omega_values[left+1]
                    right_freq=omega_values[right+1]
                    left_index=left
                    right_index=right
                    center=i

    # Calculate bandwidth (BW) and center frequency (Fc)
    elif C_band=='near':
        C_band_wavelengths=a/omega_values[a/omega_values>wavelength_cutoff] 
        i=np.where(a/omega_values==C_band_wavelengths[0])[-1][0]-1
        
        for i in range(i,len(group_index)):
            index=group_index[i]
            print(i, 'group_index =',index)
            if i>0 and i<len(group_index):
                
                max_magnitude= np.abs(index)+ (np.abs(index)*variation_threshold)
                min_magnitude = np.abs(index) - np.abs((index* variation_threshold))
        
                
                # Determine the indices for the specified frequency range
                
                left=i
                right=i
                while(np.abs(group_index[left])< max_magnitude and np.abs(group_index[left]) > min_magnitude):
                    if left>0:
                        left=left-1
                                    
                    else:
                        break
                while(np.abs(group_index[right])<max_magnitude and np.abs(group_index[right])>min_magnitude):
                    if right<len(group_index)-1:
                        right=right+1
                    else:
                        break
                
                left=left+1
                right=right-1
                
                
                print('left=',left,'center=',i,'right=',right )
                bandwidth=omega_values[left+1]-omega_values[right+1]
               
                product=np.abs(index*bandwidth)/omega_values[i+1]
                
                if product>l:
                    max_group_index=index
                    l=product
                    left_freq=omega_values[left+1]
                    right_freq=omega_values[right+1]
                    left_index=left
                    right_index=right
                    center=i

    
    elif C_band=='fixed':
        for delta in np.linspace(0,10,50):
            
           
            C_band_wavelengths=a/omega_values[a/omega_values<wavelength+delta] 
            C_band_wavelengths=C_band_wavelengths[C_band_wavelengths>wavelength-delta]   
            
            if len(C_band_wavelengths)>=1:
                
                i=np.where(a/omega_values==C_band_wavelengths[0])[-1][0]-1
                index=group_index[i]
                max_magnitude= index+ (index*variation_threshold)
                min_magnitude = index - (index* variation_threshold)
                print(i)
                    
                    # Determine the indices for the specified frequency range
                    
                left=i
                right=i
                while(group_index[left]<= max_magnitude and group_index[left] >= min_magnitude):
                    if left>0:
                        left=left-1
                                    
                    else:
                        break
                while(group_index[right]<=max_magnitude and group_index[right]>=min_magnitude):
                    if right<len(group_index)-1:
                        right=right+1
                    else:
                        break
                
                left=left+1
                if (right==len(group_index)-1 and group_index[right]<=max_magnitude and group_index[right]>=min_magnitude):
                    pass
                    
                else:
                    right=right-1
                
                print('left=',left,'center=',i,'right=',right )
                bandwidth=omega_values[left+1]-omega_values[right+1]
               
                product=np.abs(index*bandwidth)/omega_values[i+1]
            
                max_group_index=index
                l=product
                left_freq=omega_values[left+1]
                right_freq=omega_values[right+1]
                left_index=left
                right_index=right
                center=i
            
                break
    
    print('Bandwidth Product:', l)
    print('Bandwidth (BW):', l/(max_group_index*omega_values[center+1]))
    print('group_index',max_group_index)
    print('bandwidth um ' ,(-1/omega_values[left_index+1]+1/omega_values[right_index+1])*1.54/(1/omega_values[center+1]))           
    print('frequency=', omega_values[center+1])
    
    
    
    if test==True:
        
        l=-1
        
        group_velocity=np.insert(group_velocity,0,group_velocity[0])
        df=pd.DataFrame({'k':k_abs,'w':omega_values,'v':group_velocity})
        df=df.sort_values(by='w')   
        df=df.reset_index(drop=True)
        if C_band=='Full':
    
            for i,omega in enumerate(df['w']):
                
                
                vel=df['v'].loc[i]
                print(vel)
                index=1/vel
                print(i, 'group_index =',1/vel ,'v=', vel)
                
                if i>0 and i<len(df['w']):
                    
                    max_magnitude= np.abs(vel)/(1-variation_threshold)
                    
                    min_magnitude =np.abs (vel)/(1+variation_threshold)
            
                    
                    # Determine the indices for the specified frequency range
                    
                    left=i
                    right=i
                    while(np.abs(df['v'].loc[left])< max_magnitude and np.abs(df['v'].loc[left]) > min_magnitude):
                        if left>0:
                            left=left-1
                                        
                        else:
                            break
                    while(np.abs(df['v'].loc[right])<max_magnitude and np.abs(df['v'].loc[right])>min_magnitude):
                        if right<len(df['w'])-1:
                            right=right+1
                        else:
                            break
                    
                    left=left+1
                    if (np.abs(df['v'].loc[right])<max_magnitude and np.abs(df['v'].loc[right])>min_magnitude and right==len(df['w'])-1):
                        pass
                    else:
                        
                    
                        right_=right-1
                    
                    
                    print('left=',left,'center=',i,'right=',right )
                    bandwidth=df['w'].loc[left]-df['w'].loc[right]
                   
                    product=np.abs(index*bandwidth)/omega
                    
                    if product>l:
                        max_group_index=index
                        l=product
                        left_freq=df['w'][left]
                        right_freq=df['w'][right]
                        left_index=left
                        right_index=right
                        center=i
        
            
        print('Bandwidth Product:', l)
        print('Bandwidth (BW):', l/(max_group_index*df['w'][center]))
        print('group_index',max_group_index)
        print('bandwidth nm ' ,-422/left_freq+422/right_freq)

        
        
        if plot==True:
            if freq=='frequency':
                if velocity:
                    plt.plot(df['w'],df['v'])
                    plt.xlabel('frequency')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=df['w'].loc[center],linestyle='--')
                else:
                    plt.plot(df['w'],1/df['v'])
                    plt.xlabel('frequency')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=df['w'].loc[center],linestyle='--')
            
            elif freq=='wave':
            
                if velocity:
                    plt.plot(a/df['w'],df['v'])
                    plt.xlabel('wavelength')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=a/left_freq,c='g',label='bandwidth')
                    plt.axvline(x=a/right_freq,c='g')
                    plt.axvline(x=a/df['w'].loc[center],linestyle='--')
                else:
                    plt.plot(a/df['w'],1/df['v'])
                    plt.xlabel('wavelength')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=a/left_freq,c='g',label='bandwidth')
                    plt.axvline(x=right_freq,c='g')
                    plt.axvline(x=a/df['w'].loc[center],linestyle='--')
                    
            else:
            
                if velocity==False:
                    plt.scatter(df['k'],1/df['v'])
                    plt.xlabel('k')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=df['k'].loc[left_index])
                    plt.axvline(x=df['k'].loc[right_index])
                    plt.axvline(x=df['k'].loc[center],linestyle='--')
                
                else:
                    
                    plt.scatter(df['k'] ,df['v'])
                    plt.xlabel('k')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=df['k'].loc[left_index])
                    plt.axvline(x=df['k'].loc[right_index])
                    plt.axvline(x=df['k'].loc[center],linestyle='--')
            
            plt.legend()    
        
    
    else:
        if plot==True:
            if freq=='frequency':
                if velocity:
                    plt.plot(omega_values[1:],group_velocity, marker='x')
                    plt.xlabel('frequency')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=omega_values[center+1],linestyle='--')
                else:
                    plt.scatter(omega_values[1:],1/group_velocity)
                    plt.xlabel('frequency')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=left_freq)
                    plt.axvline(x=right_freq)
                    plt.axvline(x=omega_values[center+1],linestyle='--')
            
            elif freq=='wave':
                
                if velocity:
                    plt.plot(1/omega_values[1:],group_velocity)
                    plt.xlabel('wavelength')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=1/omega_values[left_index+1],c='g',label='bandwidth')
                    plt.axvline(x=1/omega_values[right_index+1],c='g')
                    plt.axvline(x=1/omega_values[center+1],linestyle='--')
                else:
                    plt.plot(1/omega_values[1:],1/group_velocity)
                    plt.xlabel('wavelength')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r',label='Group index')
                    plt.axvline(x=1/omega_values[left_index+1],c='g',label='bandwidth')
                    plt.axvline(x=1/omega_values[right_index+1],c='g')
                    plt.axvline(x=1/omega_values[center+1],linestyle='--')
                    
            else:
                
                if velocity==False:
                    plt.scatter(k_abs[1:],group_index)
                    plt.xlabel('k')
                    plt.ylabel('group index')
                    plt.axhline(y=max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=k_abs[left_index+1])
                    plt.axvline(x=k_abs[right_index+1])
                    plt.axvline(x=k_abs[center+1],linestyle='--')
                
                else:
                    
                    plt.scatter(k_abs[1:],group_velocity)
                    plt.xlabel('k')
                    plt.ylabel('velocity')
                    plt.axhline(y=1/max_group_index,alpha=0.2,c='r')
                    plt.axvline(x=k_abs[left_index+1])
                    plt.axvline(x=k_abs[right_index+1])
                    plt.axvline(x=k_abs[center+1],linestyle='--')
            plt.legend()
            
            return omega_values[1:],1/group_velocity
        
    

def get_colour_plot_data(bands,grid1,grid2):
    n1=len(grid1)
    n2=len(grid2)
    l=np.zeros((n1,n2))
    f=np.zeros((n1,n2))
    group=np.zeros(
    (n1,n2))
    for i in range(20):
        for j in range(20):
            l[i][j],f[i][j],group[i][j]=plot_band_structure(bands[i][j],22,zoom=True
                    ,freq='',plot=False
                                                        
                   )   
    return group, l ,f
    

