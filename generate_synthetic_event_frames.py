import numpy as np

def synthetic_event_frame(events, events_time, temporal_window_sec):
    dt = temporal_window_sec
    t = events_time
    t = t-t[0]

    t_init = t[0]
    t_end = t[-1]

    t_range = np.arange(t_init, t_end+dt, dt)
    t_hist,t_binEdge = np.histogram(t,bins=t_range)
    t_hist_cum = np.cumsum(t_hist)

    events_hist = np.zeros((t_hist_cum.shape[0],240,320))
    prev = 0
    curr = t_hist_cum[0]
    

    for i in range(t_hist_cum.shape[0]):
        if i == 0:
            prev = 0
            curr = t_hist_cum[i]
        else:
            prev = t_hist_cum[i-1]
            curr = t_hist_cum[i]
        max_num = 0
        # temp = events_hist[i]
        events_temp = events[prev:curr,:]

        for j in range(events_temp.shape[0]):
            x = events_temp[j,0]
            y = events_temp[j,1]

            events_hist[i,y,x] +=1

            if events_hist[i,y,x] > max_num:
                max_num = events_hist[i,y,x]

        events_hist[i] =events_hist[i]/ max_num

    return events_hist
