import numpy as np



def gauss3sigma(events):
# expects events in (n,2) format
# where events(1,2) -> (x,y) in pixels

    event_round = np.rint(events)
    event_round = event_round.astype(np.uint8)
    event_image = np.zeros((180,240))
    for i in range(events.shape[0]):
        hor = event_round[i][0]
        ver = event_round[i][1]
        if hor < 240 and ver>=0 and ver < 180 and ver>=0:
            X,Y = np.meshgrid(np.arange(max(hor-3,0), min(hor+4,239)),np.arange(max(ver-3,0), min(ver+4,179)))
            exponent = ((X-hor)**2 + (Y-ver)**2)/2
            amplitude = 1/ (np.sqrt(2*np.pi))
            val = amplitude* np.exp(-exponent)
            event_image[Y.min():Y.max()+1,X.min():X.max()+1] += val
            #event_image[Y.min().astype(int):Y.max().astype(int)+1,X.min().astype(int):X.max().astype(int)+1] += val

    contrast = -np.var(event_image)

    return contrast,event_image
    

# aa = np.array([[244,255],[26,48],[15,59]])

# contrast,img = gauss3sigma(aa)
# print(contrast)


# def gauss3sigma(events):
# # expects events in (n,2) format
# # where events(1,2) -> (x,y) in pixels

#     event_round = np.rint(events)
#     event_round = event_round.astype(np.uint8)
#     event_image = np.zeros((180,240))
#     for i in range(events.shape[0]):
#         hor = event_round[i][0]
#         ver = event_round[i][1]
#         if hor < 240 and ver>=0 and ver < 180 and ver>=0:
#             # X,Y = np.meshgrid(np.arange(max(hor-3,0), min(hor+4,239)),np.arange(max(ver-3,0), min(ver+4,179)))
#             #exponent = ((X-hor)**2 + (Y-ver)**2)/2
#             #amplitude = 1/ (np.sqrt(2*np.pi))
#             #val = amplitude* np.exp(-exponent)
#             #event_image[Y.min():Y.max()+1,X.min():X.max()+1] += val
#             event_image[ver,hor] += 1

#     contrast = -np.var(event_image)

#     return contrast,event_image
