# the ASHRAE model was chosen due to that other than the monthly
# coefficients (A, B and C) and the cosine of the solar zenith angle (cos_z),
# the model does not depend on any atmospheric data

# ASHRAE (2005) Handbook of Fundamentals, SI Edition. American Society of
# Heating, Refrigerating and Air-Conditioning Engineers, Atlanta, GA

# Bakirci, K. "Estimation of Solar Radiation by Using ASHRAE Clear-Sky Model in
# Erzurum, Turkey." Energy Sources, Part A: Recovery, Utilization, and
# Environmental Effects, vol. 31, no. 3, 2009, pp. 208-216.,
# doi:10.1080/15567030701522534.

# Abouhashish, Mohamed. "Applicability of ASHRAE Clear-Sky Model Based on
# Solar-Radiation Measurements in Saudi Arabia." 2017, doi:10.1063/1.4984509.

import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
import h5py as h5
from matplotlib.pyplot import figure
from operator import itemgetter
import sys
import pickle
import argparse

# constants that are not frame dependent
def file_constants(gloc, md, dn):

  # lat and lon coordinates are only given every nth pixel in the x and y dir
  X_step = gloc.attrs['X_Stepsize_Pixels']
  Y_step = gloc.attrs['Y_Stepsize_Pixels']
  # dimensions of each frame
  X_pixel = md[0][3] # how many pixels are in the x dir
  Y_pixel = md[0][4] # how mnay pixels are in the y dir

  cal = md[0][24] # absoluteCalCoeff_wcmsq convert image data from counts to W/cm^2
  cal = cal * 10000 # convert from cm^-2 to m^-2

  # constants that only depend on the day number
  A, B, C = constants(dn)
  ET = ET_eq(dn) # in minutes

  return(X_step, Y_step, X_pixel, Y_pixel, cal, A, B, C, ET)

def coordinate_matrix(gloc, md, counts, frame, X_step, Y_step, X_pixel,
    Y_pixel, cal, GG, framen):

  # these are the corner coordinates for creating extrapolated/(semi-interpolated) data
  corners = np.array([md[frame][41], md[frame][42], md[frame][43], md[frame][44], md[frame][45], md[frame][46], md[frame][47], md[frame][48]])
  # how many rows/cols are in the GeoLocationData
  row = int(Y_pixel / Y_step)
  col = int(X_pixel / X_step)

  cal = md[0][24] # absoluteCalCoeff_wcmsq convert image data from counts to W/cm^2
  cal = cal * 10000 # convert from cm^-2 to m^-2
  GG[framen][:][:] = counts
  counts[counts > np.percentile(counts,99)] = 0
  counts[counts < np.percentile(counts,50)] = 0
  intensity = counts * cal # Watts/m^2
  lat = np.zeros((row, col))
  lon = np.zeros((row, col))

  for i in range(row):
    lat[:][i]=list(map(itemgetter(3), gloc[frame][i][:]))
    lon[:][i]=list(map(itemgetter(4), gloc[frame][i][:]))
  if ((corners < -180).sum()):
    # nearest neighbor interpolation
    lat = np.vstack([lat, lat[-1:,:]])
    lat = np.hstack([lat, lat[:,-1:]])
    lon = np.vstack([lon, lon[-1:,:]])
    lon = np.hstack([lon, lon[:,-1:]])
    # size 36x36 array (geoloc gives lat/lon every 12th pixel)
    grid_x_orig, grid_y_orig = np.mgrid[0:X_pixel +1:int(X_pixel/row),
        0:Y_pixel +1 :int(Y_pixel/col)]
    # size 432x432 array (goal grid size)
    grid_x, grid_y = np.mgrid[0:X_pixel:1, 0:Y_pixel:1]
    points = (grid_x_orig, grid_y_orig)
    grid_x_orig = np.reshape(grid_x_orig, (row+1) * (col+1))
    grid_y_orig = np.reshape(grid_y_orig, (row+1) * (col+1))
    lon = np.reshape(lon, (row+1) * (col+1))
    lat = np.reshape(lat, (row+1) * (col+1))
    indexArr = np.argwhere(lon < -180)
    lon = np.delete(lon, indexArr)
    lat = np.delete(lat, indexArr)
    grid_x_orig = np.delete(grid_x_orig, indexArr)
    grid_y_orig = np.delete(grid_y_orig, indexArr)
    longitude = np.zeros((X_pixel,Y_pixel))
    latitude = np.zeros((X_pixel,Y_pixel))
    longitude = griddata((grid_y_orig,grid_x_orig), lon, (grid_x, grid_y), method='cubic')
    latitude = griddata((grid_y_orig,grid_x_orig), lat, (grid_x, grid_y), method='cubic')
    longitude = longitude.T
    latitude = latitude.T
  else:
    #               0       1       2       3       4       5       6       7
    # corners = [ UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon]
    # bottom row
    bot = np.linspace(corners[4], corners[6], 37)
    # take out the last col
    newrow = bot[:-1]
    # new right side column
    right = np.vstack(np.linspace(corners[2], corners[6], 37))

    # adding new row/col to lat
    lat = np.vstack([lat, newrow])
    lat = np.hstack([lat, right]) # latitudes of the image

    bot = np.linspace(corners[5], corners[7], 37)
    newrow = bot[:-1]
    right = np.vstack(np.linspace(corners[3], corners[7], 37))

    lon = np.vstack([lon, newrow])
    lon = np.hstack([lon, right]) # longitudes of the image

    def interp(x_new, y_new, x, y, z):
        f = interpolate.interp2d(x,y,z, kind = 'cubic')
        return f(x_new,y_new)

    x = np.linspace(0, X_pixel, row + 1)
    y = np.linspace(0, Y_pixel, col + 1)
    x_new = np.linspace(0, X_pixel - 1, X_pixel)
    y_new = np.linspace(0, Y_pixel - 1, Y_pixel)

    latitude = interp(x_new, y_new, x, y, lat)
    longitude = interp(x_new, y_new, x, y, lon)
  LSTM = longitude

  return latitude, longitude, LSTM, intensity, GG

# equation of time (mins)
def ET_eq(dn):
  B_n = 360 * (dn - 81) / 365
  return 9.87 * np.sin(np.deg2rad(2 * B_n)) - 7.53 * np.cos(np.deg2rad(B_n)) - 1.50 * np.sin(np.deg2rad(B_n))

# solar declination [deg]
def delta_eq(dn):
  return -np.arcsin(.39779 * np.cos(.98565 * np.pi / 180 * (304 + 10) + 1.914 * np.pi / 180 * np.sin(.98568 * np.pi/180 *(304-2) ))) * 180 / np.pi

# local solar time (LST) calculated from local standard time (LT) with equation of time (ET),
# local longitude and local standar time meridian
def LST_eq(LT, ET, LSTM, lon):
  return LT + (ET + 4*(lon - LSTM))/60

# calculates solar hour angle (H) that gives the apparent solar time for the current time period in degrees
def hr_angle_eq(LST):
  return 15 * (LST - 12)

# cosine of the zenith angle, day number, longitude, hr angle
def cos_z_eq(dn, lat, H):
  delta = delta_eq(dn)
  cosz = np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(H)) + \
         np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(lat))
  cosz = np.array(cosz)
  cosz[np.isnan(cosz)] = 1
  cosz[cosz <= 0] = .000001
  return cosz, delta

def constants(dn):

  # solar constant [W/m^2]
  I_sc = 1367

  # day angle [radians]
  Gamma = 2 * np.pi * dn / 365

  # Eccentricity correction-factor of the Earths orbit [dimensionless]
  # Accuality: ratio of the average dist between Earth and Sun and the actual dist according the dn
  E_o = 1.00011 + 0.034221 * np.cos(Gamma) + 0.00128 * np.sin(Gamma) + \
        0.000719 * np.cos(2 * Gamma) + 0.000077 * np.sin(2 * Gamma)

  # Extraterrestial radiation [W/m^2]
  A = I_sc * E_o

  B = -1.9925E-15 * dn**6 + 2.22076E-12 * dn**5 - 8.33643E-10 * dn**4 + \
      1.07543E-7 * dn**3 - 4.6E-7 * dn**2 - 131.45E-6 * dn + .14323

  C = -5.19886E-15 * dn**6 + 5.7539E-12 * dn**5 - 2.2713E-9 * dn**4 + \
      3.70022E-7 * dn**3 - 2.1351E-5 * dn**2 + 0.000511 * dn + 0.05363

  return A, B, C

# beam radiation
def I_b_eq(A, B, cosz):
  return A * np.exp(-B/cosz)

# diffuse radiation
def I_d_eq(C, I_b):
  return C * I_b

def insolation(dn, lat, lon, LSTM, LT, ET, A, B, C):
  LST = LST_eq(LT, ET, LSTM, lon) #local solar time
  H = hr_angle_eq(LST)
  cosz, delta = cos_z_eq(dn, lat, H)
  I_b = I_b_eq(A, B, cosz)
  I_d = I_d_eq(C, I_b)
  insol = I_b * cosz + I_d
  insol = np.round(insol,2)
  return(insol)

def R_norm(data, nframes):
  count = 0
  for i in range(nframes):
    R = data[i,:,:]
    R[R > np.percentile(R,99.5)] = 0
    #normalizing
    R = (R - np.min(R))/ (np.max(R) - np.min(R))
    thres = .5
    R[R <= thres] = thres
    data[i,:,:] = (R - thres) / (np.max(R) - thres)
    count = count + 1
  return data

def plotting(data, R_n, n, newcmp, savefig,indices,md):

  if n < 4:
    col = 2
    row = n
  if n == 4:
    col = 4
    row = 2
  if n == 6:
    col = 6
    row = 2
  if n == 8:
    col = 4
    row = 4
  fig, axs = plt.subplots(row,col, figsize=(20,15))

  def left(ax, colormap, data, frame, md):
    ax.matshow(data, cmap = colormap)
    # corners = [UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon]

    corners = np.array([md[frame][41], md[frame][42], md[frame][43], md[frame][44], md[frame][45], md[frame][46], md[frame][47], md[frame][48]])
    ax.text(0.0, 1.0, '[' + '%.1f' %corners[0] + ', ' + '%.1f' %corners[1] + ']',
        color='r',ha="left", va="top", transform=ax.transAxes)
    ax.text(1.0, 1.0, '[' + '%.1f' %corners[2] + ', ' + '%.1f' %corners[3] + ']',
        color='r',ha="right", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.0, '[' + '%.1f' %corners[4] + ', ' + '%.1f' %corners[5] + ']',
        color='r',ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.0, 0.0, '[' + '%.1f' %corners[6] + ', ' + '%.1f' %corners[7] + ']',
        color='r',ha="right", va="bottom",transform=ax.transAxes)
    ax.set_ylabel('\n \n Frame ' + '%d'  %frame)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

  def right(ax, colormap, data):
    im = ax.matshow(data, cmap = colormap)
    ax.axis('off')
    return(im)

  k = 0
  for i in range(row):
    for j in range(0,col,2):
      # corners = [UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon]
      left(axs[i,j], 'gray', data[k][:][:], indices[k], md)
      im = right(axs[i,j+1], newcmp, R_n[k][:][:])
      k = k + 1
  cb_ax = fig.add_axes([0.21, 0.05, 0.6, 0.02])
  cbar = fig.colorbar(im, cax=cb_ax, orientation ='horizontal' )
  cbar.set_label('probability of cloud')
  fig.tight_layout(pad=0.75)
  fig.subplots_adjust(top=0.95)
  fig.suptitle('CloudMask', fontsize=16)

def main():

  parser = argparse.ArgumentParser(description='SciTec CloudMask')
  parser.add_argument("filename", type=str, help="name of hdf5 file you want to import")
  parser.add_argument("start", help="beginning frame number", type=int)
  parser.add_argument("end", help="ending frame number", type=int)
  parser.add_argument('-s', help="flag to save images to png files", action="store_true", dest = 's', default=False)
  parser.add_argument('-n', help="chooses frames for you", action="store_true", dest = 'n', default=False)
  args = parser.parse_args()
  # load in all the lat/lon data
  f = h5.File(args.filename, 'r')
  md = f['frameMetaData']
  gloc = f['GeoLocationData']
  dset = f['CalRawData']
  # load in color map
  fp = open('cmap.pkl', 'rb')
  newcmp = pickle.load(fp)
  fp.close()
  year = md[0][15]
  dn = md[0][16] # day number
  nframes = args.end + 1 - args.start
  X_step, Y_step, X_pixel, Y_pixel, cal, A, B, C, ET = file_constants(gloc, md, dn)
  # corners
  k = 0
  indices = []
  data = np.zeros((nframes, md[0][6], md[0][7]))
  d = data.copy()
  if args.n:
    nframes = 4
    indices = [0, 1, 3, 4]
    for i in range(nframes):
      data[i][:][:] = dset[:][indices[i]][:]
  else:
    for i in range(nframes):
      indices.append(args.start + i)
      data[i][:][:] = dset[:][indices[i]][:]
  GG = np.zeros((nframes, md[0][6], md[0][7]));
  # array matrix to hold cloudmask data for each frame
  R = np.zeros((nframes, md[0][6], md[0][7]))
  for frame in range(len(indices)):
    UTC_hr = md[indices[frame]][17]/60/60 # seconds of day to hr
    latitude, longitude, LSTM, intensity, GG = coordinate_matrix(gloc, md,
        data[:][int(frame)][:], indices[frame], X_step, Y_step, X_pixel,
        Y_pixel, cal, GG, frame)
    # an estimate based off every 15 degrees being equivalent to delta UTC = 1 hr
    LT = LSTM / 15 + UTC_hr # local standard time
    Irradiance = insolation(dn, latitude, longitude, LSTM, LT, ET, A, B, C)
    Irradiance[Irradiance <= .01 ] = .001
    num = np.max(Irradiance)/10
    R[frame,:,:] = intensity/(num + Irradiance)
  R_n = R_norm(R, nframes)
  plotting(GG, R_n, nframes, newcmp, args.s, indices, md)
  plt.show()
if __name__ == '__main__':
  main()

