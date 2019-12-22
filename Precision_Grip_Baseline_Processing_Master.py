#!/usr/bin/env python
# coding: utf-8

# # Precision Grip (Baseline) Data Processing Outline

# # Import modules and set user paths

# In[1]:


## set plot output style

get_ipython().run_line_magic('matplotlib', 'qt')

## import modules

import mne
import numpy as np
import re
import os
import os.path
from pathlib import Path
from mne.report import Report
from matplotlib import pyplot as plt
from mne.preprocessing import compute_proj_ecg, compute_proj_eog, create_eog_epochs, create_ecg_epochs

## set user_path variable as directory to DataAnalysis folder in Dropbox

delim = os.path.sep

# Find the user's Home directory and set up paths to DataAnalysis folder in Dropbox
home_Path = str(Path.home())

if re.search('dylan', home_Path, re.IGNORECASE):
    user_path="/Users/dylandaniels/Dropbox (Brown)/99_shared/DataAnalysis"
elif re.search('tariq', home_Path, re.IGNORECASE):
    user_path="/Users/tariqcannonier/Dropbox/DataAnalysis"
elif re.search('simona', home_Path, re.IGNORECASE):
    user_path='C:\\Users\\Simona\\Dropbox (Brown)\\Dropbox_Work_VitalityProject\\DataAnalysis'

    
## option to test functions as you proceed through script cell by cell
year='2017'
session='pre'
test_fxn=True
print_diagnostic=False
test_report=False


# In[2]:


def set_directories_vitality (DataAnalysis_path, year, session, print_diagnostic = False ):
    
    ## define subdirectories in relation to DataAnalysis using OS path delimited 'delim'
    
    data_path=delim+year+' Vitality EEG Analysis'+delim+'Precision Grip'+delim+session+delim+'EEG_EMG'+delim     +'1_Grip_'+session+'_raw_set'+delim+'Grip_'+session+'_All'+delim
    
    output_path=delim+year+' Vitality EEG Analysis'+delim+'Precision Grip'+delim+session+delim+'EEG_EMG'+delim     + '2_Grip_PRE_MNE_processed' +delim
    
    report_path=delim+year+' Vitality EEG Analysis'+delim+'reports'+delim+session+delim
    
    montage_path=delim+year+' Vitality EEG Analysis'+delim+'MATLAB script'+delim+session+delim
    
    ## define directories from subdirectories
    
    data_directory = DataAnalysis_path + data_path
    output_directory = DataAnalysis_path + output_path
    montage_directory = DataAnalysis_path + montage_path
#     report_directory = DataAnalysis_path + report_path
    report_directory = '/Users/tariqcannonier/Jones_Lab/Precision_Grip' + report_path
    
    ## get filenames
    
    data_filenames = [f for f in os.listdir(data_directory)                          if f.endswith('.set')] # list .set files in data directory
    
    ## optionally print directories
    
    if print_diagnostic == True:
        print('\n###\n### Printing data directory ... \n###\n\n', data_directory, "\n\n",               '\n###\n### Printing output directory ... \n###\n\n', output_directory, "\n\n",               '\n###\n### Printing report directory ... \n###\n\n', report_directory, "\n\n",               '\n###\n### Printing data filenames ... \n###\n\n', data_filenames, "\n")
        
    return data_directory, output_directory, report_directory, data_filenames;

#test_fxn=True

if test_fxn==True: # test function
    data_directory, output_directory, report_directory,     data_filenames = set_directories_vitality (user_path, year, session, print_diagnostic)


# In[3]:


## define function to get participant_info dictionary from file_list

def get_data_info( file_list, # file_list is a list of .set files to analyze \
                  data_dir, # filepath to data directory \
                  output_directory, #filepath to output directory \
                  print_diagnostic = False ): # value of True prints participant_info
    
    participants=[]
    inpaths=[]
    outpaths=[]
    
    for e in file_list:
        
        ## get participant number from filename
        pnum = e.split("_")[0] # grabs contents of filename before first underscore
        participants+=[pnum] # saves string with participant number to list
        
        ## set input path 
        inpaths+=[data_dir+e] # set the input 
        
        ## create new output name
        outname = e.split("AllChannels.set")[0]
        outname+='mne_processed.set'
        
        ## set output path
        outpaths+=[output_directory+outname]

    ## create dictionary with participant info
    
    # participant_info = {'ID': (input_path, output_path), ...}
    participant_info={}
    index=0
    for i in range(0,len(participants)):
        participant_info[participants[i]] = inpaths[i],outpaths[i]
        
    ## optionally print dictionary with participant info
    if print_diagnostic==True: # print participant_info
        print('\n###\n### Printing \'participant_info\' dictionary ... \n###\n\n----------\n')
        for key, value in participant_info.items(): 
            print('Participant:',key,'\n\nInpath:',value[0],'\n\nOutpath:',value[1],'\n\n----------\n')

    return participant_info; # return dictionary with participant info


#test_fxn=True

if test_fxn==True: # test function
    participant_info = get_data_info( data_filenames, data_directory, output_directory, print_diagnostic)


# # Import and filter data; view data properties

# In[4]:


##### define function as preprocess_mydata
### also separate out emg channels

## Define function to save and filter EEG channels

def filter_mydata( input_path , filter_params):
        
    ## import raw data; preload into memory
    raw_data = mne.io.read_raw_eeglab(input_path, preload=True)
        
    ## copy raw data
    working_data = raw_data.copy() 

    ## rename E
    working_data.rename_channels({'E':'STI 014'}) 
    working_data.set_channel_types({'STI 014':'stim'}) 
    
    ## Separate out eeg and emg channels
    emg_only = working_data.copy().pick_channels(['T7', 'T8', 'PO7', 'PO8'])
    eeg_only = working_data.copy().pick_channels(['Fp1', 'Fp2', 'F3', 'Fz', 'F4',                                                   'C3', 'Cz', 'C4', 'P3', 'Pz',                                                   'P4', 'Oz'])   
        
    ## filter EEG channels
    eeg_only.filter(filter_params['eeg']['highpass'],filter_params['eeg']['lowpass'],                     fir_design='firwin',verbose=False)
    
    ## set EMG bipolar reference
    mne.set_bipolar_reference(emg_only,['T7' , 'PO7'], ['T8' , 'PO8'],                               ch_name=['T7-T8' , 'PO7-PO8'],drop_refs=False,copy=False)
    
    ## set EEG bipolar reference
    mne.set_bipolar_reference(eeg_only,['C3' , 'C4'], ['Cz' , 'Cz'],                              ch_name=['C3-Cz' , 'C4-Cz'],drop_refs=False,copy=False)

    ## highpass EMG channels
    emg_only.filter(filter_params['emg']['highpass'],None,fir_design='firwin',verbose=False)
    
    ## rectify EMG data
    emg_only.apply_function(np.absolute)
    
    ## low pass rectified EMG data
    emg_only.filter(None,filter_params['emg']['lowpass'],fir_design='firwin',verbose=False)
    
    ## plot "envelope"??

    
    ## package EEG and EMG data in dictionary
    filtered_data = {'eeg':eeg_only,'emg':emg_only}

    return raw_data, working_data, filtered_data; # emg_only # return filtered data

if test_fxn==True: # test function

    highpass_eeg = 0.01 # set high-pass filter
    lowpass_eeg = 50. # set low-pass filter
    highpass_emg = 20.
    lowpass_emg = 100.
    filter_params = {'eeg':{'highpass':highpass_eeg, 'lowpass':lowpass_eeg},                      'emg':{'highpass':highpass_emg,'lowpass':lowpass_emg}}
    
    participant_ID='2010' # set participant to analyze 

    raw_data, working_data,    filtered_data = filter_mydata( participant_info[participant_ID][0] , filter_params )


# # View data properties, plot channels

# In[5]:


## define function to view data properties

def view_data_properties ( list_properties, # list of properties in ".info" to view 
                          data_file ):
    
    if type(list_properties) != list:
        print('TypeError: the list_properties argument must be a list.              \n\nNote: an empty list "[]" will return the value of data_file.info().')
    
    ## print the specified properties
    elif type(list_properties) == list and list_properties != []:
        print("\n-----\n")
        for e in list_properties:
            print(str(e),":",data_file.info[e],"\n")
        print("-----\n")
        
    ## if no properties are select, print the entirety of ".info"
    elif type(list_properties) == list and list_properties==[]:
        print(data_file.info)

    return

if print_diagnostic==True: # test function
    print_props=['ch_names','bads','highpass','lowpass','sfreq'] # set data properties to view
    #print_props=[]

    view_data_properties ( print_props , working_data )
    view_data_properties ( print_props , filtered_data['eeg'] )
    view_data_properties ( print_props , filtered_data['emg'])


# In[6]:


## function to plot channels

def plot_channels(plot_list):
    
    ## track which item from plot_list is being plotted
    fig_text="Figure " # to be referenced below
    count=1
    
    ## loop through plotlist and plot channels
    for e in plot_list:
        
        # plot elements of list, mindful of dictionaries
        if type(e) == type(dict()):
            
            for key in e.keys():
                fig_label=fig_text+str(count) # set figure label from count

                print('\n-----\n\nPloting',str(e[key]),"as ",fig_label,"...\n") # print item info to output

                # Put a title on psd plots
                fig = e[key].plot_psd(average=False,xscale='linear');
                ax = fig.get_axes()
                ax[0].set_title('%s Channels' % key.upper())
                count+=1
                
        else:
            fig_label=fig_text+str(count) # set figure label from count

            print('\n-----\n\nPloting',str(e),"as ",fig_label,"...\n") # print item info to output
            
            # Put a title on psd plots
            fig = e.plot_psd(average=False,xscale='linear'); # generate plot; semicolon suppresses duplicate plots
            ax = fig.get_axes()
            ax[0].set_title('ALL Channels')
            count+=1
        
    return

# raw.set_eeg_reference('average', projection=True)  # set EEG average reference
if print_diagnostic==True: # test function

#     plot_list=[filtered_data['eeg'],filtered_data['emg']]

    plot_channels([(working_data.copy().pick_types(eeg=True,emg=False)) , 
                   filtered_data ])
    #plot_channels(plot_list)


# # Epoching

# **Our data comes from EEGLAB and so we will need to use [events_from_annotations()](https://www.nmr.mgh.harvard.edu/mne/stable/generated/mne.events_from_annotations.html) command to get events from the data format EEGLAB exports**

# In[7]:


## epoch data by block timestamps

def epoch_data ( data_file, print_diagnostic = False ): #define function
    
    ## identify events
    events, event_id = mne.events_from_annotations(data_file) # get events from data in EEGLAB format
    for key in event_id.keys(): # iterate through event_id keys to provide meaningful annotations
        if key == '100.0':
            event_id['StartBlock'] = event_id.pop('100.0') # annotate 100 as startblock
        if key == '200.0':
            event_id['EndBlock'] = event_id.pop('200.0') # annotate 200 as endblock
        
    ## generate array of timestamps
    timestamps=[] # create list to hold startblock and endblock times

    # Compare events to get timestamps.  Only look at consecutive StartBlocks and EndBlocks
    prev_event = np.array([0,0,0])
    for event in events:
        if event[2] == 2 and prev_event[2] == 1:
            block_timestamp = [prev_event[0],event[0]]
            timestamps += [block_timestamp]
        prev_event=event
        
    timestamps=np.asarray(timestamps) # convert timestamps list into array
    
    ## optionally print and plot events
    if print_diagnostic == True:
        print('\n###\n### Printing event IDs ... \n###\n\n',event_id,'\n\n')
        print('\n###\n### Printing events ... \n###\n\n',events,'\n\n')
        print('\n###\n### Printing timestamps ... \n###\n\n',timestamps,'\n')
        events_fig = mne.viz.plot_events(events, sfreq=data_file.info['sfreq']);
        
        # Title and legend in figure
        ax = events_fig.get_axes()
        ax[0].set_title('Events in Continuous Data')
        ax[0].legend(['StartBlock','Endblock','255'])
    
    return timestamps, events;

if test_fxn==True: # test function
    # leaving out EMG processing until we know how we want to crop emg data
    epochs, events = epoch_data( filtered_data['eeg'] , print_diagnostic) # run epoching function
    


# # Crop data; process events and epochs

# In[8]:


def crop_data (data_file, timestamps, print_diagnostic = False ):
    
    id_label=1
    event_duration=2
    
    # Instantiate dict for epoched data
    filtered_blocks = {
        'eeg':{'blocks':[],'events':[],'epochs':[]},\
        'emg':{'blocks':[],'events':[],'epochs':[]}
    }

    # Iterate through timestamps and derive times to crop data
    for time in timestamps:
        tmin = time[0]/data_file['eeg'].info['sfreq']
        tmax = time[1]/data_file['eeg'].info['sfreq']
#         print("---\n",tmin,"\n\n",tmax,"\n")

        # Iterate through crop and epoch EEG and EMG data
        for key in data_file.keys():

            filtered_blocks[key]['blocks'].append(data_file[key].copy().crop                                (tmin=tmin,tmax=tmax)) # return a list of eeg lab arrays split into blocks by timestamps
            filtered_blocks[key]['events'].append(mne.make_fixed_length_events                                  (filtered_blocks[key]['blocks'][-1],id=id_label,duration=event_duration)) # for each block, return a list of arrays with event markers 0,1 every 2s

            # need to rename this to be event marked_block or marked_data; blocks_epochs is not accurate
            filtered_blocks[key]['epochs'].append(mne.Epochs                                  (filtered_blocks[key]['blocks'][-1],filtered_blocks[key]['events'][-1], event_id=id_label,                                   tmin=0,tmax=2, baseline=None,                                   preload=True,verbose=False)) # add 2s [0,1] event markers to each block in array
        
    if print_diagnostic==True:
        print('\n###\n### Printing all epochs for each block ... \n###\n\n',filtered_blocks['emg']['events'],"\n\n-----\n") 

    return filtered_blocks;

if test_fxn==True: # test function
    # leaving out EMG processing until we know how we want to crop emg data
    filtered_blocks = crop_data(filtered_data , epochs, print_diagnostic)


# In[9]:


def generate_block_figs(filtered_blocks, n_epochs, duration, scalings, i):

    plt.ioff() # turns off plots
    block_figs = {'eeg':None,'emg':None}
    
    # define figures for report
    for key in filtered_blocks.keys():
        
        # Select channels if you're iterating through EEG or EMG data
        if key == 'eeg':
            picks = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'Oz']
        else:
            picks = ['T7', 'T8', 'PO7', 'PO8']
        
        plot_blocks = filtered_blocks[key]['blocks'][i].plot(events=filtered_blocks[key]['events'][i],                                                          duration=duration,show=False,scalings=scalings);
        butterfly = filtered_blocks[key]['epochs'][i].average().plot(show=False,scalings=scalings);
        ## PROBABLY CAN REMOVE THE TOPOMAP FOR EMG    
        topomap = filtered_blocks[key]['epochs'][i].copy().pick(picks).average().plot_topomap(show=False,                                                                                              scalings=scalings);
        topojoint = filtered_blocks[key]['epochs'][i].copy().pick(picks).average().plot_joint(show=False);
        
        # save report figures to list
        block_figs.update( { key: [ plot_blocks, butterfly, topomap, topojoint ] } )
    
    return block_figs;

#test_fxn=True

if test_fxn==True: # test function
    n_epochs = 3 # Use for viewing subset of epochs.
    duration = n_epochs*2 # Use for viewing subset of epochs. Otherwise set to 40
    scalings = 1/25000 # Setting it to a constant to compare artifact in epoching
    i = 1
    block_figs = generate_block_figs(filtered_blocks, n_epochs, duration, scalings, i);


# # Report

# In[10]:


## define report

def reporting(epochs, subject_ID, filtered_blocks,              working_data, filtered_data, events, report_directory):
    
    # Define EEG and EMG explicitly.  Can use this solution until mne patches pick_types(emg=True)
    eeg_chans = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'Oz']
    emg_chans = ['T7', 'T8', 'PO7', 'PO8']
    
    plt.ioff() # turns off plots
            
    rep = Report() # call Report object
        
    # Plot continuous data plot
    eeg_chs_raw = working_data.copy().pick_channels(eeg_chans).plot_psd(average=False,xscale='linear', show=False);
    eeg_chs_filtered = filtered_data['eeg'].plot_psd(average=False,xscale='linear', show=False);
    emg_chs_raw = working_data.copy().pick_channels(emg_chans).plot_psd(average=False,xscale='linear', show=False);
    emg_chs_filtered = filtered_data['emg'].plot_psd(average=False,xscale='linear', show=False);
    show_events = mne.viz.plot_events(events, sfreq=working_data.info['sfreq'], show=False);
    
    partic_figs=[eeg_chs_raw, eeg_chs_filtered, emg_chs_raw, emg_chs_filtered, show_events]
    captions = ["raw EEG psd","filtered EEG psd","raw EMG psd","filtered EMG psd","events"]
    
    # Put title in figure for continuous data plot
    for e, c in zip(partic_figs,captions):
        ax = e.get_axes()
        ax[0].set_title(c)
            
    # Add caption to report page
    rep.add_figs_to_section(partic_figs, captions=["raw EEG psd","filtered EEG psd",                                                   "raw EMG psd","filtered EMG psd","events"],                            section="Subject "+subject_ID)
    
    for i in range(0,len(epochs)): # loop through blocks
        
        # make figures
        block_figs = generate_block_figs(filtered_blocks, n_epochs, duration, scalings, i);

        # Iterate through EEG and EMG
        for key in block_figs.keys():
            
            # define figure captions
            captions = ['Block %d Data %s' % (i+1, key.upper()),                     'Block %d Butterfly %s' % (i+1, key.upper()),                     'Block %d Topomap %s' % (i+1, key.upper()),                     'Block %d TopoJoint %s' % (i+1, key.upper())]


            # add list of figures to report
            rep.add_figs_to_section( figs=block_figs[key], captions=captions, section='Subject '+subject_ID+' Block %d' % (i+1))
            
            
    # set report filename
    filename = os.getcwd()+os.path.sep+subject_ID+'_'+session+'_report.html'

    # save report
    rep.save(filename, overwrite=True, open_browser=False)
    
    return;
    
if test_report==True: # test function
    participant_ID='2010'
    
    # leaving out EMG processing until we know how we want to crop emg data
    reporting(epochs, participant_ID, filtered_blocks, working_data,              filtered_data, events, report_directory)
    


# # Loop through multiple subjects

# In[11]:


## code to loop through subjects and generate reports

def run_reports( subject_list, filter_params, n_epochs, duration, scalings ):
    
    plt.ioff() # turns off plots

    data_directory, output_directory, report_directory, data_filenames =     set_directories_vitality (user_path, year, session, False)
    
    participant_info = get_data_info( data_filenames, data_directory, output_directory, False )
    
    for e in subject_list: # loop through subjects, set input path
        
        subject_ID=e
        
        if e in participant_info.keys():
            
            print( 'Starting Participant %s' % subject_ID)        

            raw_data, working_data, filtered_data = filter_mydata( participant_info[e][0] , filter_params )

            epochs, events = epoch_data( filtered_data['eeg'], False)

            # leaving out EMG processing until we know how we want to crop emg data
            filtered_blocks = crop_data(filtered_data , epochs, False)

            reporting(epochs, subject_ID, filtered_blocks,                     working_data, filtered_data, events, report_directory);
            


# In[ ]:





# # Workflow for looping through subjects

# In[12]:


## define list of subject(s) to analyze
subject_list=['2004', '2010', '2037']
# subject_list=['2037', '2004', '2025', '2012', '2016', '2021', '2029', '2017',\
#               '2020', '2024', '2013', '2028', '2046', '2042', '2032', '2001',\
#               '2036', '2009', '2019', '2026', '2011', '2015', '2022', '2038',\
#               '2003', '2034', '2030', '2007', '2039', '2031', '2002', '2045',\
#               '2041', '2018', '2014', '2023', '2010']

## set analysis properties
highpass_eeg = 0.01 
lowpass_eeg = 50.
highpass_emg = 20. 
lowpass_emg = 100.
filter_params = {'eeg':{'highpass':highpass_eeg, 'lowpass':lowpass_eeg},                  'emg':{'highpass':highpass_emg,'lowpass':lowpass_emg}}
mne.set_log_level(verbose=False) # Quiets the outputs from functions

## set report properties
n_epochs = 20
duration = n_epochs*2
scalings = 1/25000

## set group
year='2017'
session='pre'

## run workflow on subject_list
run_reports( subject_list, filter_params, n_epochs, duration, scalings );


# In[ ]:




