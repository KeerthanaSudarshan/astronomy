#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import savgol_filter
from PyAstronomy import pyasl
import math
import datetime
import pandas as pd
import astropy.units as u
import astropy.time
import astropy.coordinates
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
plt.rcParams["figure.facecolor"] = 'white'


# # Coordinates conversion

# In[178]:


def to_altaz(date, time):
    
    objects = ['M26', 'Alya', 'Deneb el Okab', 'M71', 'Albireo', 'Delta Cygni', 'B 296', '?']
    ra_array = [281.65, 284.35, 286.63, 298.71, 292.92, 296.43, 271.02, 275.55]
    dec_array = [-9.36, 4.23, 13.90, 18.84, 28, 45.19, -24.52, -13.96]
    gal_lon = [25, 35, 45, 55, 65, 75, 5.89, 17.21]
    gal_lat = [0,0, 0, 0, 0, 0, -1.324, -0.0086]
    sonipat = EarthLocation(lat=28.9931*u.deg, lon=77.0151*u.deg, height=224*u.m)
    utcoffset = +5.5*u.hour #Eastern Daylight Time
    datetime = '2023-07-'+str(date) + ' ' + time+ ':00'
    time_utc = Time(datetime) - utcoffset
    alt = []
    az = []
    
    for i in range(0,len(objects)):
        ra = ra_array[i]
        dec = dec_array[i]
        
        loc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        
        altaz = loc.transform_to(AltAz(obstime=time_utc, location=sonipat))
        alt.append(f'{altaz.alt:.6}')
        az.append(f'{altaz.az:.6}')
        
    data = {'Name' : objects, 'Az' : az, 'Alt' : alt, 'G Lat (deg)' : gal_lat, 'G Long (deg)' : gal_lon}
    return pd.DataFrame(data)


# In[136]:


def radec_to_altaz(ra_deg, dec_deg, date_str, time_str, lat=28.9931, lon=77.0151, elev=224, utc_offset_hours=5.5):
    obj_radec = SkyCoord(ra=ra_deg*u.deg, dec= dec_deg*u.deg, frame='icrs')
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
    utcoffset = utc_offset_hours*u.hour #Eastern Daylight Time
    time = Time(date_str+' '+time_str) - utcoffset
    obj_altaz = obj_radec.transform_to(AltAz(obstime=time, location=loc))
    return f'alt= {obj_altaz.alt:.6}', f'az= {obj_altaz.az:.6}'

def altaz_to_radec(alt_deg, az_deg, date_str, time_str, lat=28.9931, lon=77.0151, elev=224, utc_offset_hours=5.5):
    loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
    utcoffset = utc_offset_hours*u.hour
    time = Time(date_str+' '+time_str) - utcoffset
    obj_altaz = SkyCoord(AltAz(obstime=time, az=az_deg*u.deg, alt=alt_deg*u.deg, location=loc))
    obj_radec = obj_altaz.transform_to('icrs')
    return obj_radec.to_string('hmsdms')


# # Plot every file in a folder
# Some small customizations possible

# In[217]:


def plot_folder(folder_name, software, low_x=None, high_x=None, scatter=True, smooth=True, baseline_sub=False):
    if software=='sdr':
        extn = 'txt'
    else:
        extn = 'csv'
    for name in sorted(glob.glob('./Data/Horn_data/'+folder_name+'/*.'+extn)):
        if software=='sdr':
            freq, p = np.loadtxt(name, delimiter='  ', unpack=True, skiprows=1)
        elif software=='rtl':
            p = np.loadtxt(name, delimiter=',', usecols=range(2,519))
            freq = np.arange(p[0], p[1]+p[2], p[2])/1e6
            p = 10**(p[4:]/10)
            background_file = input('Enter background file for file '+name.split('/')[-1]+': ')
            if background_file!='':
                p_background = np.loadtxt('./Data/Horn_data/'+folder_name+'/'+background_file, delimiter=',', usecols=range(6,519))
                p_background = 10**(p_background/10)
                p = p/max(p)
                p_background = p_background/max(p_background)
                p = p/p_background
        elif software=='gnu':
            freq, p = np.loadtxt(name, delimiter=',', unpack=True)
            
        p = p[low_x:high_x]
        freq = freq[low_x:high_x]
        if baseline_sub:
            p = (p - savgol_filter(p, window_length=(len(p)//2)*2-1, polyorder=1))
        if scatter:
            plt.scatter(freq, p, s=5, color='blue')
        if smooth:
            p_list = p.tolist()
            peak = freq[int(p_list.index(max(p)))]
            full_label = 'Peak: '+str(np.round(peak,3))+' MHz'
            plt.plot(freq, savgol_filter(p, window_length=31, polyorder=2), color='red', label=full_label)
            plt.legend()
        plt.xlabel('Frequency (MHz)', fontsize=14)
        plt.ylabel('Power', fontsize=14)
        plt.title(name.split('/')[-1])
#         plt.ylim(5.50e-5,7e-5)
#         plt.ylim(0.0045, 0.0076)
        plt.show()


# # Fully customizing plots

# ### Frequency plots

# In[299]:


def freq_plot(file_name, software, scatter=True, smooth=True, baseline_sub=True, label=None, line=False, fig=None, ax=None, low_x=200, high_x=450, color=None):
    if fig==None:
        fig,ax=plt.subplots()
    if software=='sdr':
        freq, p = np.loadtxt(file_name, delimiter='  ', unpack=True, skiprows=1)
        units='(Arbitrary units)'
    elif software=='rtl':
        p = np.loadtxt(file_name, delimiter=',', usecols=range(2,519))
        freq = np.arange(p[0], p[1]+p[2], p[2])/1e6
        p = 10**(p[4:]/10)
        background_file = input('Enter background file for file '+file_name.split('/')[-1]+': ')
        if background_file!='':
            p_background = np.loadtxt('./Data/Horn_data/'+file_name.split('/')[-2]+'/'+background_file, delimiter=',', usecols=range(6,519))
            p_background = 10**(p_background/10)
            p = p/max(p)
            p_background = p_background/max(p_background)
            p = p/p_background
        units='(watts)'
    elif software=='gnu':
        freq, p = np.loadtxt(file_name, delimiter=',', unpack=True)
        units='(Kelvin?)'
        
    ax.set_xlabel('Frequency (MHz)', fontsize=14)
    ax.set_ylabel('Power '+units, fontsize=14)
    freq_cut = freq[low_x:high_x]
    p_cut = p[low_x:high_x]
    if baseline_sub:
        p_cut = p_cut - savgol_filter(p_cut, window_length=(len(p_cut)//2)*2-1, polyorder=1)
    p_list = p_cut.tolist()
    peak = freq_cut[int(p_list.index(max(p_cut)))]
    full_label = label+': '+str(np.round(peak,3))+' MHz'
    if scatter:
        ax.scatter(freq_cut, p_cut, s=5, color=color, alpha=0.5, label = full_label)
        full_label=None
    if smooth:
        ax.plot(freq_cut, savgol_filter(p_cut, window_length=31, polyorder=2), label = full_label, color=color)
    if line:
        ax.axvline(x=peak, ls='--', color='black')
    ax.legend(fontsize=12, loc=2, markerscale=3)
    ax.ticklabel_format(useOffset=False)
    return


# ### VLSR Correction and velocity graphs

# In[139]:


def lsr(coords_array, date_str, time_str, lat=28.9931, lon=77.0151, elev=224, utc_offset_hours=5.5):
    az_, alt_ = coords_array
#     month, day, hour, minute, second = time_array
    azimuth = az_
    altitude = alt_
    time = date_str+' '+time_str
    sonipat = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=elev*u.m)
    utcoffset = utc_offset_hours*u.hour 
    time = Time(time) - utcoffset
    new_altaz = SkyCoord(AltAz(obstime=time, az=azimuth*u.deg, alt=altitude*u.deg, location=sonipat))

    newradec = new_altaz.transform_to('icrs')

    longitude = lon
    latitude = lat
    altitude = elev
    obs_ra_2000, obs_dec_2000 = newradec.ra.deg, newradec.dec.deg 
    date_split = np.array(date_str.split('-')).astype(int)
    time_split = np.array(time_str.split(':')).astype(int)
    dt = datetime.datetime(date_split[0],date_split[1],date_split[2], time_split[0],time_split[1],time_split[2])
    jd = pyasl.jdcnv(dt)
    corr, hjd = pyasl.helcorr(longitude, latitude, altitude, obs_ra_2000, obs_dec_2000, jd, debug=False)
    v_sun = 20.5 
    sun = get_sun(Time(time))
    sun_icrs = sun.transform_to('icrs')
    sun_ra = math.radians(sun_icrs.ra.deg)
    sun_dec = math.radians(sun_icrs.dec.deg)
    obs_dec = math.radians(obs_dec_2000)
    obs_ra = math.radians(obs_ra_2000)

    a = math.cos(sun_dec) * math.cos(obs_dec)
    b = (math.cos(sun_ra) * math.cos(obs_ra)) + (math.sin(sun_ra) * math.sin(obs_ra))
    c = math.sin(sun_dec) * math.sin(obs_dec)
    v_rs = v_sun * ((a * b) + c)
    v_lsr = corr + v_rs
    return -v_lsr


# In[140]:


def vel_plot(file_name, software, coords_array, date_str, time_str, scatter=True, smooth=True, baseline_sub=True, label=None, line=False, fig=None, ax=None, low_x=200, high_x=450, color=None):
    if fig==None:
        fig,ax=plt.subplots()
    if software=='sdr':
        freq, p = np.loadtxt(file_name, delimiter='  ', unpack=True, skiprows=1)
        units='(Arbitrary units)'
    elif software=='rtl':
        p = np.loadtxt(file_name, delimiter=',', usecols=range(2,519))
        freq = np.arange(p[0], p[1]+p[2], p[2])/1e6
        p = 10**(p[4:]/10)
        background_file = input('Enter background file for file '+file_name.split('/')[-1]+': ')
        if background_file!='':
            p_background = np.loadtxt('./Data/Horn_data/'+file_name.split('/')[-2]+'/'+background_file, delimiter=',', usecols=range(6,519))
            p_background = 10**(p_background/10)
            p = p/max(p)
            p_background = p_background/max(p_background)
            p = p/p_background
        units='(watts)'
    elif software=='gnu':
        freq, p = np.loadtxt(file_name, delimiter=',', unpack=True)
        units='(Kelvin?)'
        
    c = 299792458*1e-3
    vel = -(freq-1420.4058)*c/1420.4058 + lsr(coords_array, date_str, time_str)
    
    ax.set_xlabel('Velocity with LSR correction (km/s)', fontsize=14)
    ax.set_ylabel('Power '+units, fontsize=14)
    vel_cut = vel[low_x:high_x]
    p_cut = p[low_x:high_x]
    if baseline_sub:
        p_cut = p_cut - savgol_filter(p_cut, window_length=(len(p_cut)//2)*2-1, polyorder=1)
    p_list = p_cut.tolist()
    peak = vel_cut[int(p_list.index(max(p_cut)))]
    full_label = label+': '+str(np.round(peak,3))+' km/s'
    if scatter:
        ax.scatter(vel_cut, p_cut, s=5, color=color, alpha=0.5, label = full_label)
        full_label=None
    if smooth:
        ax.plot(vel_cut, savgol_filter(p_cut, window_length=31, polyorder=2), label = full_label, color=color)
    if line:
        ax.axvline(x=peak, ls='--', color='black')
    ax.legend(fontsize=12, loc=1, markerscale=3)
    ax.ticklabel_format(useOffset=False)
    return

