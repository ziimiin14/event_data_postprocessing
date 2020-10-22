import numpy as np

time_vesc = np.fromfile('../vesc_bin_file/VESC_6300ERPM_Time.bin',dtype=np.float64)
time_event = np.fromfile('../event_data_14102020/6300ERPM_Time_14102020.bin',dtype=np.int64)
vesc = np.fromfile('../vesc_bin_file/VESC_6300ERPM_RPM.bin',dtype=np.float64)
event = np.fromfile('../event_data_14102020/6300ERPM_Events_14102020.bin',dtype=np.uint8)


# time_opti = np.fromfile('data_set1/6300ERPM_Max_Opti_TimeMod.bin',dtype=np.int64)
# time_event = np.fromfile('data_set1/6300ERPM_Max_TimeMod.bin',dtype=np.int64)
# opti = np.fromfile('data_set1/6300ERPM_Max_Opti_QuatMod.bin',dtype=np.float64)
# event = np.fromfile('data_set1/6300ERPM_Max_EventMod.bin',dtype=np.uint8)

time_vesc = time_vesc.reshape(int(time_vesc.shape[0]),1)
time_event = time_event.reshape(int(time_event.shape[0]),1)
vesc = vesc.reshape(int(vesc.shape[0]/2),2)
event = event.reshape(int(event.shape[0]/3),3)


time_vesc = time_vesc *1e9
time_interval = (1/100)*1e9

# Event preprocess before rotation started
time_init = time_event[0]
time_end = time_event[-1]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_event,bins=time_range)
time_hist_cum = np.cumsum(time_hist)


check1 = np.where(time_hist>3000)
event = event[time_hist_cum[check1[0][0]-1]:]
time_event = time_event[time_hist_cum[check1[0][0]-1]:]

# Event preprocess again relative to VESC
time_vesc = time_vesc.astype('int64')
vesc = vesc.astype('int64')
time_event = time_event - time_event[0]

check2 = np.where(time_event[:,0]<time_vesc[-1])
time_event = time_event[check2[0]]
event = event[check2[0]]

vesc_final = np.concatenate((time_vesc,vesc),axis=1)

vesc_final_mod = np.zeros((vesc_final.shape[0]+1,vesc_final.shape[1]),dtype=np.int64)
vesc_final_mod[1:,:] = vesc_final
vesc_final_mod = np.delete(vesc_final_mod,1,1)

print(vesc_final_mod.shape,vesc_final_mod.dtype)
print(event.shape,event.dtype)
print(time_event.shape,time_event.dtype)

vesc_final_mod.tofile('../vesc_event_syn_data/8820ERPM_VescMod_14102020.bin')
event.tofile('../vesc_event_syn_data/8820ERPM_EventsMod_14102020.bin')
time_event.tofile('../vesc_event_syn_data/8820ERPM_TimeMod_14102020.bin')
# time_opti.tofile('data_set1/6300ERPM_Max_Opti_TimeMod.bin')
# time_event.tofile('data_set1/6300ERPM_Max_TimeMod.bin')
# event.tofile('data_set1/6300ERPM_Max_EventMod.bin')
# quat.tofile('data_set1/6300ERPM_Max_Opti_QuatMod.bin')
# result = np.append(time_mod,quat,axis=1)
# result.tofile('6300ERPM_2000_Opti_New.bin')
