import matplotlib.pyplot as plt

def spike_raster_plot(t,data,colors=None):
    if colors is None:
        colors = ['blue']
    if data.ndim > 1:
        for neuron in range(len(data)):
            spike_locs = []
            for step in range(len(t)):
                if data[neuron][step] > 0:
                    spike_locs.append(t[step])
            if len(colors) == 1:
                plt.eventplot(spike_locs,lineoffsets=neuron+1, colors=colors[0],linelengths=0.8)
            else:
                plt.eventplot(spike_locs, lineoffsets=neuron + 1, colors=colors[neuron], linelengths=0.8)
    else:
        spike_locs = []
        for step in range(len(t)):
            if data[step] > 0:
                spike_locs.append(t[step])
        plt.eventplot(spike_locs, lineoffsets=1, colors=colors[0], linelengths=0.8)