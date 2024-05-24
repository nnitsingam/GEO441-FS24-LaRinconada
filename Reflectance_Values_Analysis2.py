# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:25:15 2024

@author: Valu
"""

from rasterio.merge import merge
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import geopandas as gpd
import pyproj
#import gdal
from matplotlib.colors import ListedColormap
import seaborn as sns
from rasterio.transform import from_origin
import matplotlib
import fiona
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import matplotlib.patches as mpatches
import matplotlib as mpl
#%%

b = rasterio.open(r"data/outputs/landsat_1999_LC_RF.tif")

AOI = gpd.read_file(r"data/input_data/Histogram_aois.shp")
mining_areas = AOI.iloc[1:2]
bare_rock = AOI.iloc[2:3]
barren_land = AOI.iloc[4:5]
#geoms = AOI.geometry.values
# mining_areas = [geom.__geo_interface__ for geom in mining_areas]
# barren_land = [geom.__geo_interface__ for geom in barren_land]
# bare_rock = [geom.__geo_interface__ for geom in bare_rock]
# #geoms = [geom.__geo_interface__ for geom in geoms]

data_arrays=[]
date_array = []

for filename in os.listdir(r"data/outputs"):
    if filename.endswith('.tif'):
        with rasterio.open(os.path.join(r"data/outputs",filename)) as src:
            src_array = src.read(fill_value = np.nan)
            data_arrays.append(src_array)
            raster_crs = src.crs
            
data_stack = np.stack(data_arrays)

top_left_x = b.transform.c
top_left_y = b.transform.f
src_transform = rasterio.transform.from_origin(top_left_x, top_left_y, b.transform[0], b.transform[0])  
src_crs = raster_crs
mining_mask = geometry_mask(mining_areas.geometry, out_shape=(data_stack.shape[2], data_stack.shape[3]), transform=src_transform, invert=True, all_touched = True)

# Apply mask to data stack
cropped_data_stack = np.empty_like(data_stack)
for i in range(data_stack.shape[0]):
    cropped_data_stack[i, :, :] = np.where(mining_mask, data_stack[i, :, :], np.nan)
    
full_image = cropped_data_stack[0,0,:,:] # this needs to be selected manually

nan_rows = np.all(np.isnan(full_image), axis=(1))
#nan_rows = nan_rows[0,0,:]
nan_rows = ~nan_rows #invert the rows
nan_cols = np.all(np.isnan(full_image), axis=(0))
#nan_cols = nan_cols[0,0,:]
nan_cols = ~nan_cols

test_data_stack = cropped_data_stack

#get the boundaries of rows and columns that can be removed
first_true_index_rows = np.argmax(nan_rows)
last_true_index_rows = len(nan_rows) - np.argmax(nan_rows[::-1]) -1

first_true_index_cols = np.argmax(nan_cols)
last_true_index_cols = len(nan_cols) - np.argmax(nan_cols[::-1]) -1

# now remove the empty columns
data_stack_mining = test_data_stack[:,:,first_true_index_rows:last_true_index_rows, first_true_index_cols:last_true_index_cols]

plt.figure()
plt.imshow(data_stack_mining[0,0,:,:])

most_frequent_values = []
for i in range(data_stack_mining.shape[0]):
    values = data_stack_mining[i,6,:,:]
    flattened_values =values.flatten()
    unique, counts = np.unique(flattened_values, return_counts = True)
    value_df = np.column_stack((unique, counts))
    most_frequent = np.argmax(value_df[:,1])
    landcover_type = value_df[most_frequent, 0]
    data_stack_mining[i,6,:,:] = landcover_type
    
    
    
barren_mask = geometry_mask(barren_land.geometry, out_shape=(data_stack.shape[2], data_stack.shape[3]), transform=src_transform, invert=True, all_touched = True)

# Apply mask to data stack
cropped_data_stack = np.empty_like(data_stack)
for i in range(data_stack.shape[0]):
    cropped_data_stack[i, :, :] = np.where(barren_mask, data_stack[i, :, :], np.nan)
    
full_image = cropped_data_stack[0,0,:,:] # this needs to be selected manually

nan_rows = np.all(np.isnan(full_image), axis=(1))
#nan_rows = nan_rows[0,0,:]
nan_rows = ~nan_rows #invert the rows
nan_cols = np.all(np.isnan(full_image), axis=(0))
#nan_cols = nan_cols[0,0,:]
nan_cols = ~nan_cols

test_data_stack = cropped_data_stack

#get the boundaries of rows and columns that can be removed
first_true_index_rows = np.argmax(nan_rows)
last_true_index_rows = len(nan_rows) - np.argmax(nan_rows[::-1]) -1

first_true_index_cols = np.argmax(nan_cols)
last_true_index_cols = len(nan_cols) - np.argmax(nan_cols[::-1]) -1

# now remove the empty columns
data_stack_barren = test_data_stack[:,:,first_true_index_rows:last_true_index_rows, first_true_index_cols:last_true_index_cols]

plt.figure()
plt.imshow(data_stack_barren[0,0,:,:])

most_frequent_values = []
for i in range(data_stack_barren.shape[0]):
    values = data_stack_barren[i,6,:,:]
    flattened_values =values.flatten()
    unique, counts = np.unique(flattened_values, return_counts = True)
    value_df = np.column_stack((unique, counts))
    most_frequent = np.argmax(value_df[:,1])
    landcover_type = value_df[most_frequent, 0]
    data_stack_barren[i,6,:,:] = landcover_type



rock_mask = geometry_mask(bare_rock.geometry, out_shape=(data_stack.shape[2], data_stack.shape[3]), transform=src_transform, invert=True, all_touched = True)

# Apply mask to data stack
cropped_data_stack = np.empty_like(data_stack)
for i in range(data_stack.shape[0]):
    cropped_data_stack[i, :, :] = np.where(rock_mask, data_stack[i, :, :], np.nan)
    
full_image = cropped_data_stack[0,0,:,:] # this needs to be selected manually

nan_rows = np.all(np.isnan(full_image), axis=(1))
#nan_rows = nan_rows[0,0,:]
nan_rows = ~nan_rows #invert the rows
nan_cols = np.all(np.isnan(full_image), axis=(0))
#nan_cols = nan_cols[0,0,:]
nan_cols = ~nan_cols

test_data_stack = cropped_data_stack

#get the boundaries of rows and columns that can be removed
first_true_index_rows = np.argmax(nan_rows)
last_true_index_rows = len(nan_rows) - np.argmax(nan_rows[::-1]) -1

first_true_index_cols = np.argmax(nan_cols)
last_true_index_cols = len(nan_cols) - np.argmax(nan_cols[::-1]) -1

# now remove the empty columns
data_stack_rock = test_data_stack[:,:,first_true_index_rows:last_true_index_rows, first_true_index_cols:last_true_index_cols]

plt.figure()
plt.imshow(data_stack_rock[0,0,:,:])

most_frequent_values = []
for i in range(data_stack_rock.shape[0]):
    values = data_stack_rock[i,6,:,:]
    flattened_values =values.flatten()
    unique, counts = np.unique(flattened_values, return_counts = True)
    value_df = np.column_stack((unique, counts))
    most_frequent = np.argmax(value_df[:,1])
    landcover_type = value_df[most_frequent, 0]
    data_stack_rock[i,6,:,:] = landcover_type

#%%
rows = []

# Iterate over each source image
for source in range(data_stack_mining.shape[0]):
    # Extract the values for the second position (1st axis)
    values = data_stack_mining[source, :, :, :]

    # Flatten the 2D values array (11, 13) into a single row (143 elements per column)
    flattened_values = values.reshape(values.shape[0], -1)

    # Create a row for each flattened value set
    for flat_value_set in flattened_values.T:
        row = [source] + list(flat_value_set)
        rows.append(row)

# Create the DataFrame
columns = ['source'] + [f'Band {i+1}' for i in range(data_stack_mining.shape[1])]
df_mining = pd.DataFrame(rows, columns=columns)
df_mining = df_mining.drop(columns=df_mining.columns[-1])

rows = []

# Iterate over each source image
for source in range(data_stack_barren.shape[0]):
    # Extract the values for the second position (1st axis)
    values = data_stack_barren[source, :, :, :]

    # Flatten the 2D values array (11, 13) into a single row (143 elements per column)
    flattened_values = values.reshape(values.shape[0], -1)

    # Create a row for each flattened value set
    for flat_value_set in flattened_values.T:
        row = [source] + list(flat_value_set)
        rows.append(row)

# Create the DataFrame
columns = ['source'] + [f'Band {i+1}' for i in range(data_stack_barren.shape[1])]
df_barren = pd.DataFrame(rows, columns=columns)
df_barren = df_barren.drop(columns=df_mining.columns[-1])


rows = []

# Iterate over each source image
for source in range(data_stack_rock.shape[0]):
    # Extract the values for the second position (1st axis)
    values = data_stack_rock[source, :, :, :]

    # Flatten the 2D values array (11, 13) into a single row (143 elements per column)
    flattened_values = values.reshape(values.shape[0], -1)

    # Create a row for each flattened value set
    for flat_value_set in flattened_values.T:
        row = [source] + list(flat_value_set)
        rows.append(row)

# Create the DataFrame
columns = ['source'] + [f'Band {i+1}' for i in range(data_stack_rock.shape[1])]
df_rock = pd.DataFrame(rows, columns=columns)
df_rock = df_rock.drop(columns=df_rock.columns[-1])
#%%


columns_to_plot = ['Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6']

# Melt the DataFrame
melted_df_mining = pd.melt(df_mining, id_vars=['source', 'Band 7'], value_vars=columns_to_plot,
                    var_name='value_type', value_name='value')
#melted_df = melted_df[(melted_df['source'] >=11) & (melted_df['source'] <=14) ]

column_values = df_mining['Band 7']
#fill_values = column_values[14::15]
unique_values = melted_df_mining['Band 7'].unique()

#palette = {value: sns.color_palette("husl", len(unique_values))[i] for i, value in enumerate(unique_values)} 
#color_values = ['#dedede', '#8e9391', '#8e9391','#ba7bde', '#8e9391', '#ba7bde', '#dedede','#ffaa00','#dedede', '#8e9391', '#ba7bde', '#ffaa00','#dedede', '#ba7bde', '#8e9391', '#dedede', '#dedede','#8e9391', '#dedede', '#dedede', '#8e9391','#dedede', '#8e9391', '#8e9391','#8e9391' ]
plt.figure(figsize=(100, 8))
sns.violinplot(x='value_type', y='value', hue='source', data=melted_df_mining, 
               split=False)
plt.show()





#%%
years = list(range(1999, 2024))

default_font_size = mpl.rcParams['font.size']

# Increase the font size by a relative factor
relative_font_size_factor = 1.2  # Increase by 20%
new_font_size = default_font_size * relative_font_size_factor

# Set the new default font size for all elements
mpl.rcParams.update({'font.size': new_font_size})
titles = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(200, 60))
g = sns.FacetGrid(melted_df_mining, col='value_type', col_wrap=3, height=4, dropna=True)

for ax, title in zip(g.axes.flat, titles):
    ax.set_title(title)
    
for ax in g.axes.flat:
#     ax.axvspan(2.5, 3.5, color='red', alpha=0.1)
#     ax.axvspan(6.5, 7.5, color='red', alpha=0.1)
#     ax.axvspan(7.5, 8.5, color='red', alpha=0.1)
#     ax.axvspan(12.5, 13.5, color='red', alpha=0.1)
#     ax.axvspan(10.5, 11.5, color='red', alpha=0.1)
#     # if 'Band 1' in ax.get_title():
#     #     ax.text(13.2, ax.get_ylim()[1] + 1800, 'Barren Land', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
#     #     ax.text(8.2, ax.get_ylim()[1] + 1800, 'Barren Land', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
#     #     ax.text(19.2, ax.get_ylim()[1] + 2500, 'Mining Area (Land)', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
    
     ax.set_xticks(range(len(years)))
     ax.set_xticklabels(years, rotation=0)
     # Label only every fifth year
     for label in ax.get_xticklabels():
         if int(label.get_text()) % 5 != 0:
             label.set_visible(False)
        
g.map_dataframe(sns.violinplot, x='source', y='value', hue='source', split=False, inner='quart', color = '#ba7bde')
#g.set_titles('{col_name}')
g.set_axis_labels('Year', 'Surface Reflectance')
# legend_handles = [
#     mpatches.Patch(color='#dedede', label='Barren Land'),
#     mpatches.Patch(color='#91d474', label='Vegetation'),
#     mpatches.Patch(color='#ba7bde', label='Mining Area (Land)'),  # Add more as needed
#     mpatches.Patch(color='#8e9391', label='Bare Rock')
# ]

# Add the custom legend to the plot
#plt.legend(handles=legend_handles, title='Classification Result', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
#plt.savefig('mining_hist.png', dpi=500, bbox_inches='tight')
plt.show()




#%%
# Melt the DataFrame
melted_df_barren = pd.melt(df_barren, id_vars=['source', 'Band 8'], value_vars=columns_to_plot,
                    var_name='value_type', value_name='value')
#melted_df = melted_df[(melted_df['source'] >=11) & (melted_df['source'] <=14) ]

column_values = df_barren['Band 8'] 

years = list(range(1999, 2024))

# default_font_size = mpl.rcParams['font.size']

# # Increase the font size by a relative factor
# relative_font_size_factor = 0.7  # Increase by 20%
#new_font_size = default_font_size * relative_font_size_factor

# Set the new default font size for all elements
#mpl.rcParams.update({'font.size': new_font_size})

plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(200, 60))
g = sns.FacetGrid(melted_df_barren, col='value_type', col_wrap=3, height=4, dropna=True)
for ax in g.axes.flat:
#     ax.axvspan(2.5, 3.5, color='red', alpha=0.1)
#     ax.axvspan(6.5, 7.5, color='red', alpha=0.1)
#     ax.axvspan(7.5, 8.5, color='red', alpha=0.1)
#     ax.axvspan(12.5, 13.5, color='red', alpha=0.1)
#     ax.axvspan(10.5, 11.5, color='red', alpha=0.1)
#     # if 'Band 1' in ax.get_title():
#     #     ax.text(13.2, ax.get_ylim()[1] + 1800, 'Barren Land', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
#     #     ax.text(8.2, ax.get_ylim()[1] + 1800, 'Barren Land', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
#     #     ax.text(19.2, ax.get_ylim()[1] + 2500, 'Mining Area (Land)', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
    
     ax.set_xticks(range(len(years)))
     ax.set_xticklabels(years, rotation=0)
     # Label only every fifth year
     for label in ax.get_xticklabels():
         if int(label.get_text()) % 5 != 0:
             label.set_visible(False)
        
g.map_dataframe(sns.violinplot, x='source', y='value', hue='source', split=False, inner='quart', color = '#dedede')
g.set_titles('{col_name}')
g.set_axis_labels('Year', 'Surface Reflectance')
# legend_handles = [
#     mpatches.Patch(color='#dedede', label='Barren Land'),
#     mpatches.Patch(color='#91d474', label='Vegetation'),
#     mpatches.Patch(color='#ba7bde', label='Mining Area (Land)'),  # Add more as needed
#     mpatches.Patch(color='#8e9391', label='Bare Rock')
# ]

# Add the custom legend to the plot
#plt.legend(handles=legend_handles, title='Classification Result', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
#plt.savefig('hist_mining_area2002.png', dpi=500, bbox_inches='tight')
plt.show()

#%%
melted_df_rock = pd.melt(df_rock, id_vars=['source', 'Band 7'], value_vars=columns_to_plot,
                    var_name='value_type', value_name='value')
#melted_df = melted_df[(melted_df['source'] >=11) & (melted_df['source'] <=14) ]

column_values = df_rock['Band 7'] 

years = list(range(1999, 2024))

# default_font_size = mpl.rcParams['font.size']

# # Increase the font size by a relative factor
# relative_font_size_factor = 0.7  # Increase by 20%
#new_font_size = default_font_size * relative_font_size_factor

# Set the new default font size for all elements
#mpl.rcParams.update({'font.size': new_font_size})

plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(200, 60))
g = sns.FacetGrid(melted_df_rock, col='value_type', col_wrap=3, height=4, dropna=True)
for ax in g.axes.flat:
#     ax.axvspan(2.5, 3.5, color='red', alpha=0.1)
#     ax.axvspan(6.5, 7.5, color='red', alpha=0.1)
#     ax.axvspan(7.5, 8.5, color='red', alpha=0.1)
#     ax.axvspan(12.5, 13.5, color='red', alpha=0.1)
#     ax.axvspan(10.5, 11.5, color='red', alpha=0.1)
#     # if 'Band 1' in ax.get_title():
#     #     ax.text(13.2, ax.get_ylim()[1] + 1800, 'Barren Land', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
#     #     ax.text(8.2, ax.get_ylim()[1] + 1800, 'Barren Land', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
#     #     ax.text(19.2, ax.get_ylim()[1] + 2500, 'Mining Area (Land)', color='black', ha='center', va='center', rotation=90, fontsize=11, bbox=dict(facecolor='none', edgecolor='none'))
    
     ax.set_xticks(range(len(years)))
     ax.set_xticklabels(years, rotation=0)
     # Label only every fifth year
     for label in ax.get_xticklabels():
         if int(label.get_text()) % 5 != 0:
             label.set_visible(False)
        
g.map_dataframe(sns.violinplot, x='source', y='value', hue='source', split=False, inner='quart', color = '#8e9391')
g.set_titles('{col_name}')
g.set_axis_labels('Year', 'Surface Reflectance')
# legend_handles = [
#     mpatches.Patch(color='#dedede', label='Barren Land'),
#     mpatches.Patch(color='#91d474', label='Vegetation'),
#     mpatches.Patch(color='#ba7bde', label='Mining Area (Land)'),  # Add more as needed
#     mpatches.Patch(color='#8e9391', label='Bare Rock')
# ]

# Add the custom legend to the plot
#plt.legend(handles=legend_handles, title='Classification Result', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
#plt.savefig('hist_mining_area2002.png', dpi=500, bbox_inches='tight')
plt.show()

