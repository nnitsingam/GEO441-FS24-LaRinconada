# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:44:34 2024

@author: Valu
"""

import rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_origin
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import setp
import numpy as np
from datetime import datetime
import pandas as pd
import geopandas as gpd

#%%

# import polygons for vegetated areas 
aoi = gpd.read_file(r"data/NDVI_timeseries/AOI_NDVI.shp")
ext1 = [0,1,5,6,7,8]
ext2 = [2,3,4]

# read data
data_stack_dict = {}
data_count = {}
for i in range(0, len(aoi)):
    if i in ext1:
        aoi_tmp = aoi.iloc[i]
        #this is used to transform other images to full extent
        b =  rasterio.open(r"data/landsat_images/PE_LX-sr_30m_2022_composite_filt.tif") 
        data_arrays =[] #will store extent info
        date_array = [] #will store date info
        clipped_arrays = []
        for filename in os.listdir(f"data\landsat_images"):
            if filename.endswith('.tif') and len(filename.split("_")) == 6:
                print(filename)
                with rasterio.open(os.path.join(f"data\\landsat_images", filename)) as src:
                    window = rasterio.windows.from_bounds(*b.bounds,transform=b.transform)
                    src_array = src.read(window=window, boundless=True,fill_value=np.nan) 
                    data_arrays.append(src_array)
                    clipped_array, _ = rio_mask(src, [aoi_tmp.geometry], crop=True,filled=True)
                    clipped_arrays.append(clipped_array)
                    date_str = filename.split("_")[3] 
                    date_array.append(datetime.strptime(date_str,"%Y")) 
        # Stack the data arrays
        data_stack_dict[f"data_stack_{i}"] = np.stack(clipped_arrays)
        
    if i in ext2:
        aoi_tmp = aoi.iloc[i]
        b = rasterio.open(r"data/NDVI_timeseries/Landsat_ext2/PE_LX-sr_30m_2022_composite_filt.tif")
        
        data_arrays =[] #will store extent info
        date_array = [] #will store date info
        clipped_arrays = []
        for filename in os.listdir(f"data/NDVI_timeseries/Landsat_ext2"):
            if filename.endswith('.tif') and len(filename.split("_")) == 6:
                print(filename)
                with rasterio.open(os.path.join(f"data/NDVI_timeseries/Landsat_ext2", filename)) as src:
                    window = rasterio.windows.from_bounds(*b.bounds,transform=b.transform)
                    src_array = src.read(window=window, boundless=True,fill_value=np.nan) 
                    data_arrays.append(src_array)
                    clipped_array, _ = rio_mask(src, [aoi_tmp.geometry], crop=True,filled=True)
                    clipped_arrays.append(clipped_array)
                    date_str = filename.split("_")[3] 
                    date_array.append(datetime.strptime(date_str,"%Y")) 
        # Stack the data arrays
        data_stack_dict[f"data_stack_{i}"] = np.stack(clipped_arrays)
        
#%% Calculate NDVI
ndvi_dict = {}
for key, value in data_stack_dict.items():
    nir = value[:,3,:,:].astype(float)  
    red = value[:,2,:,:].astype(float)
    
    mask = (nir + red) != 0  # Create a mask where both NIR and Red are not zero
    ndvi = np.zeros_like(nir, dtype=np.float32)  # Initialize NDVI array
    ndvi[mask] = (nir[mask] - red[mask]) / (nir[mask] + red[mask])  # Calculate NDVI where mask is True
    
    # Handle division by zero: set NDVI to NaN where mask is False
    ndvi[~mask] = np.nan   
    
    ndvi = ndvi[:, np.newaxis, :, :]  # Reshape NDVI to have the same shape as other bands
    modified_array = np.concatenate((value,ndvi),axis=1)
    ndvi_dict[key] = modified_array
    
#%% Calculate EVI
evi_dict = {}
for key, value in ndvi_dict.items():
    nir = value[:,3,:,:].astype(float)  
    red = value[:,2,:,:].astype(float)
    blue = value[:,0,:,:].astype(float)
    
    evi = np.zeros_like(nir, dtype=np.float32)
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    evi = evi[:,np.newaxis,:,:]
    modified_array = np.concatenate((value,evi), axis=1)
    evi_dict[key] = modified_array
    
 #%% Prepare T and P data for plotting
# import temperature data
temperature = pd.read_csv(r"data/NDVI_timeseries/mean_annual_T.csv", sep=',')
# get the years
temperature['system:time_start'] = pd.to_datetime(temperature['system:time_start'])
# Extract the year from the dates and add it as a new column
temperature['year'] = temperature['system:time_start'].dt.year

# get the corresponding years
temperature = temperature[temperature['year'] >= 1984]
temperature.reset_index(drop=True, inplace=True)
#temperature.head
    
# Import rain data
precipitation = pd.read_csv(r"data/NDVI_timeseries/mean_annual_P.csv", sep=';')
precipitation['system:time_start'] = pd.to_datetime(precipitation['system:time_start'])
precipitation['year'] = precipitation['system:time_start'].dt.year
precipitation = precipitation[precipitation['year'] >= 1984]
precipitation.reset_index(drop=True, inplace= True)
#precipitation.head    
    
#%% Plot NDVI (test)
ndvi_selected_scene = ndvi_dict['data_stack_3'][38, 6, :, :]
print(date_array[38])
# Plot NDVI
plt.figure(figsize=(10, 8))
plt.imshow(ndvi_selected_scene, cmap='viridis', vmin=0.5, vmax=0.7)  # Set the colormap and range as per your preference
plt.colorbar(label='NDVI')
plt.title('NDVI of Selected Scene')
plt.xlabel('Pixel Column')
plt.ylabel('Pixel Row')
plt.show()  
  
#%%

plt.plot(precipitation['undefined'])
plt.show()  
    
plt.plot(temperature['undefined'])
plt.show

#%% create timeseries plot for maximal NDVI values and for each AOI
ndvi_max_list = []
for key, value in ndvi_dict.items():
    ndvi = value[:,6,:,:]
    ndvi_max = np.nanmax(ndvi, axis=(1,2))
    ndvi_max_list.append(ndvi_max)

ndvi_max_df = pd.DataFrame(ndvi_max_list)
ndvi_max_df = ndvi_max_df.transpose()

# rename columns
ndvi_max_df.columns = [f'ndvi_max_{i}' for i in range(ndvi_max_df.shape[1])]

year_labels = [date.year for date in date_array]
climate_df = pd.DataFrame({
    't': temperature['undefined'],
    'p': precipitation['undefined'],
    'year': year_labels
})

df = pd.concat([ndvi_max_df, climate_df], axis=1)
df.sort_values(by='year')   

#%% all in one
fig, axes = plt.subplots(3,1,figsize=(10,10), sharex=True)
axes[0].plot(df['year'], df.iloc[:,0], linestyle= 'dotted', color='darkgreen', label='AOI 0')
axes[0].plot(df['year'], df.iloc[:,1], linestyle= 'dashdot', color='darkgreen', label='AOI 1')
axes[0].plot(df['year'], df.iloc[:,5], linestyle= 'dotted', color='limegreen', label='AOI 10')
axes[0].plot(df['year'], df.iloc[:,6], linestyle= 'dashdot', color='limegreen', label='AOI 11')
axes[0].plot(df['year'], df.iloc[:,7], linestyle= 'dashed', color='limegreen', label='AOI 12')
axes[0].plot(df['year'], df.iloc[:,8], linestyle= 'solid', color='limegreen', label='AOI 12')
axes[0].plot(df['year'], df.iloc[:,2], linestyle= 'dotted', color='olive', label='AOI 2')
axes[0].plot(df['year'], df.iloc[:,3], linestyle= 'dashdot', color='olive', label='AOI 3')
axes[0].plot(df['year'], df.iloc[:,4], linestyle= 'dashed', color='olive', label='AOI 4')
#axes[0].set_ylim(0.3, 0.9)
axes[0].legend()
#axes[0].set_xlim(1984.0,2023.0)
#axes[0].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[0].set_xlabel('Year')
axes[0].set_ylabel('NDVI')
axes[0].set_title('Yearly maximum NDVI for vegetated 3x3 pixel areas in site 1')

axes[1].plot(df['year'], df['t'], color='red')
#axes[1].set_ylim(2, 6)
#axes[1].set_xlim(-1, 38)
#year_lab = temperature['year']
#axes[1].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Temperature [°C]')
axes[1].set_title('Mean annual air temperature')

axes[2].plot(df['year'], df['p'], color='blue')
#axes[2].set_ylim(1400, 2000)
#year_lab = precipitation['year']
#axes[2].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Precipitation [mm]')
axes[2].set_title('Mean annual precipitation')


plt.tight_layout()
plt.show()

#%% for each extent
default_font_size = mpl.rcParams['font.size']

# Increase the font size by a relative factor
relative_font_size_factor = 1.7  # Increase by 20%
new_font_size = default_font_size * relative_font_size_factor

# Set the new default font size for all elements
mpl.rcParams.update({'font.size': new_font_size})



plt.rcParams['font.family'] = 'Arial'

fig, axes = plt.subplots(5,1,figsize=(18,20), sharex=True)


axes[0].plot(df['year'], df.iloc[:,8], linestyle= 'solid', color='limegreen', label='AOI 1')
axes[0].plot(df['year'], df.iloc[:,7], linestyle= 'dashed', color='limegreen', label='AOI 2')
axes[0].plot(df['year'], df.iloc[:,6], linestyle= 'dashdot', color='limegreen', label='AOI 3')
axes[0].plot(df['year'], df.iloc[:,5], linestyle= 'dotted', color='limegreen', label='AOI 4')

axes[1].plot(df['year'], df.iloc[:,0], linestyle= 'dotted', color='darkgreen', label='AOI 5')
axes[1].plot(df['year'], df.iloc[:,1], linestyle= 'dashdot', color='darkgreen', label='AOI 6')



axes[2].plot(df['year'], df.iloc[:,2], linestyle= 'dotted', color='olive', label='AOI 7')
axes[2].plot(df['year'], df.iloc[:,3], linestyle= 'dashdot', color='olive', label='AOI 8')
axes[2].plot(df['year'], df.iloc[:,4], linestyle= 'dashed', color='olive', label='AOI 9')

axes[0].set_ylim(0.4, 0.85)
axes[1].set_ylim(0.4, 0.85)
axes[2].set_ylim(0.4, 0.85)

axes[0].legend()
axes[1].legend()
axes[2].legend()
#axes[0].set_xlim(1984.0,2023.0)
#axes[0].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[0].set_xlabel('Year')
axes[0].set_ylabel('NDVI')
axes[0].set_title('Yearly maximum NDVI for a vegetated 3x3 pixel areas in site 1')

axes[1].set_xlabel('Year')
axes[1].set_ylabel('NDVI')
axes[1].set_title('Yearly maximum NDVI for a vegetated 3x3 pixel areas in site 2')

axes[2].set_xlabel('Year')
axes[2].set_ylabel('NDVI')
axes[2].set_title('Yearly maximum NDVI for a vegetated 3x3 pixel areas in site 3')

axes[3].plot(df['year'], df['t'], color='red')
#axes[1].set_ylim(2, 6)
#axes[1].set_xlim(-1, 38)
#year_lab = temperature['year']
#axes[1].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[3].set_xlabel('Year')
axes[3].set_ylabel('Temperature [°C]')
axes[3].set_title('Mean annual air temperature')

axes[4].plot(df['year'], df['p'], color='blue')
#axes[2].set_ylim(1400, 2000)
#year_lab = precipitation['year']
#axes[2].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[4].set_xlabel('Year')
axes[4].set_ylabel('Precipitation [mm]')
axes[4].set_title('Mean annual precipitation')
# background color
axes[0].axvspan(2001, 2010, color='lightgrey', alpha=0.2)
axes[1].axvspan(2001, 2010, color='lightgrey', alpha=0.2)
axes[2].axvspan(2001, 2010, color='lightgrey', alpha=0.2)
axes[3].axvspan(2001, 2010, color='lightgrey', alpha=0.2)
axes[4].axvspan(2001, 2010, color='lightgrey', alpha=0.2)

plt.tight_layout()
plt.show()

#%% create timeseries plot for mean EVI values and for each AOI
evi_mean_list = []
for key, value in evi_dict.items():
    evi = value[:,7,:,:]
    evi_mean = np.nanmean(evi, axis=(1,2))
    evi_mean_list.append(evi_mean)

evi_mean_df = pd.DataFrame(evi_mean_list)
evi_mean_df = evi_mean_df.transpose()

# rename columns
evi_mean_df.columns = [f'evi_mean_{i}' for i in range(evi_mean_df.shape[1])]

df_evi = pd.concat([evi_mean_df, climate_df], axis=1)
df_evi.sort_values(by='year')   

evi_max_list = []
for key, value in evi_dict.items():
    evi = value[:,7,:,:]
    evi_max = np.nanmax(evi, axis=(1,2))
    evi_max_list.append(evi_max)

evi_max_df = pd.DataFrame(evi_max_list)
evi_max_df = evi_max_df.transpose()

# rename columns
evi_max_df.columns = [f'evi_max_{i}' for i in range(evi_max_df.shape[1])]

df_evi_max = pd.concat([evi_max_df, climate_df], axis=1)
df_evi_max.sort_values(by='year')  
    
#%% plot evi

fig, axes = plt.subplots(3,1,figsize=(10,10), sharex=True)
axes[0].plot(df_evi['year'], df_evi.iloc[:,0], color='lawngreen', label='AOI 0')
axes[0].plot(df_evi['year'], df_evi.iloc[:,1], color='limegreen', label='AOI 1')
axes[0].plot(df_evi['year'], df_evi.iloc[:,2], color='green', label='AOI 2')
axes[0].plot(df_evi['year'], df_evi.iloc[:,3], color='darkgreen', label='AOI 3')
axes[0].plot(df_evi['year'], df_evi.iloc[:,4], color='olive', label='AOI 4')
#axes[0].set_ylim(0.3, 0.9)
axes[0].legend()

#axes[0].set_xlim(1984.0,2023.0)
#axes[0].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[0].set_xlabel('Year')
axes[0].set_ylabel('EVI')
axes[0].set_title('Yearly mean EVI for a vegetated 3x3 pixel area')

axes[1].plot(df_evi['year'], df_evi['t'], color='red')
#axes[1].set_ylim(2, 6)
#axes[1].set_xlim(-1, 38)
#year_lab = temperature['year']
#axes[1].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Temperature [°C]')
axes[1].set_title('Mean annual air temperature')

axes[2].plot(df_evi['year'], df_evi['p'], color='blue')
#axes[2].set_ylim(1400, 2000)
#year_lab = precipitation['year']
#axes[2].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Precipitation [mm]')
axes[2].set_title('Mean annual precipitation')

plt.tight_layout()
# setp(axes[1].get_xticklabels(), visible=True) # doesnt work in matplotlib 2.2
#plt.set_tick_params(which='both', labelbottom=True)
plt.show()
    
    
  # plot max evi
fig, axes = plt.subplots(3,1,figsize=(10,10), sharex=True)
axes[0].plot(df_evi_max['year'], df_evi_max.iloc[:,0], color='lawngreen', label='AOI 0')
axes[0].plot(df_evi_max['year'], df_evi_max.iloc[:,1], color='limegreen', label='AOI 1')
axes[0].plot(df_evi_max['year'], df_evi_max.iloc[:,2], color='green', label='AOI 2')
axes[0].plot(df_evi_max['year'], df_evi_max.iloc[:,3], color='darkgreen', label='AOI 3')
axes[0].plot(df_evi_max['year'], df_evi_max.iloc[:,4], color='olive', label='AOI 4')
#axes[0].set_ylim(0.3, 0.9)
axes[0].legend()

#axes[0].set_xlim(1984.0,2023.0)
#axes[0].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[0].set_xlabel('Year')
axes[0].set_ylabel('EVI')
axes[0].set_title('Yearly mean EVI for a vegetated 3x3 pixel area')

axes[1].plot(df_evi_max['year'], df_evi_max['t'], color='red')
#axes[1].set_ylim(2, 6)
#axes[1].set_xlim(-1, 38)
#year_lab = temperature['year']
#axes[1].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Temperature [°C]')
axes[1].set_title('Mean annual air temperature')

axes[2].plot(df_evi_max['year'], df_evi_max['p'], color='blue')
#axes[2].set_ylim(1400, 2000)
#year_lab = precipitation['year']
#axes[2].set_xticks(ticks=np.arange(0,len(year_labels),5), labels=year_labels[::5])
axes[2].set_xlabel('Year')
axes[2].set_ylabel('Precipitation [mm]')
axes[2].set_title('Mean annual precipitation')

# background color
axes[1].axvspan(2001, 2010, color='lightgrey', alpha=0.5)

plt.tight_layout()
# setp(axes[1].get_xticklabels(), visible=True) # doesnt work in matplotlib 2.2
#plt.set_tick_params(which='both', labelbottom=True)
plt.show()  
    
    
    
    