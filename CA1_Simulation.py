from __future__ import division
import dipoleWrapperNeuron
import poissonSourceAxon
from numpy import *
from matplotlib.pyplot import *
from random import randint, gauss
from scipy.stats import poisson
import pickle
from scipy.signal import butter, lfilter, freqz
from matplotlib.font_manager import FontProperties, findfont
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# Set up time
tau = 0.1
maxTime = 2.0
maxTimeInMS = maxTime * 1000
maxTimeStep = maxTime / tau
tspan = arange(0,maxTimeInMS/tau, 1)

# Condition
primed = True

# Set any pharmacological manipulations
ACh = False
Retigabine = False
PCP = False
PCPPartial = False

# Uncomment the label you want for the saved plots, indicating which pharmacological manipulations were present.
pharmas = 'Baseline'
# pharmas = 'PCP'
# pharmas = 'Retigabine'
# pharmas = 'PCPPlusRetigabine'
# pharmas = 'ACh'
# pharmas = 'AChPerSE'
# pharmas = 'noEPrime'
# pharmas = 'noIPrime'

# Neuron Parameters
neuronHeight = 0.00056

# Input Parameters
predictionInputCount = 200
sensoryInputCount = 200
ratioOfCorrelatedInhibition = 0.5
predictionEOnset = 300
predictionIOnset = 320
predictionEOffset = 1800
predictionIOffset = 1820
sensoryEOnset = 100
sensoryEOffset = 1800
sensoryIOnset = 120
sensoryIOffset = 1820

# Input Weights
predictionEWeight = 15
predictionIWeight = 12.5 #10
sensoryEWeight = 4 #4
sensoryIWeight = 18 #18

# Input Lambdas
offLambda = 0.0001
predictionELambda = 0.01
predictionILambda = 0.01
if primed:    
    sensoryELambda = 0.01
    sensoryILambda = 0.01
    primedTerm = "Primed"
else:
    sensoryELambda = 0.0001
    sensoryILambda = 0.0001
    primedTerm = "Unprimed"

correlatedPredictionCount = int(sensoryInputCount * ratioOfCorrelatedInhibition)
unCorrelatedPredictionCount = sensoryInputCount - correlatedPredictionCount

correlatedSensoryCount = int(predictionInputCount * ratioOfCorrelatedInhibition)
unCorrelatedSensoryCount = predictionInputCount - correlatedSensoryCount

# Set up the neuron
neuron = dipoleWrapperNeuron.DipoleWrapperNeuron(inputs_a=[], inputs_b=[], outputs=[], externalInput_a=0.0, externalInput_b=0.0, distance = neuronHeight, position = [randint(1,100), randint(1,100), 0.0], debug=False, tau=tau, tspan=tspan, inactivating=False)

if ACh:
    neuron.base.gdapt = 3.2
    neuron.apex.gdapt = 3.2

if PCP:
    neuron.base.g_nmda = 1.8e-3
    neuron.apex.g_nmda = 1.8e-3
    sensoryILambda = 0.001
    predictionILambda = 0.001

if Retigabine:
    neuron.base.gdapt = 5.0
    neuron.apex.gdapt = 5.0
    sensoryIWeight += 6.0

predictionEConnections = []
predictionIConnections = []
sensoryEConnections = []
sensoryIConnections = []

# Set up prediction input
# for n in range(sensoryInputCount):
#     weight = predictionEWeight * gauss(1.0, 0.1) #Original weight 0.025
#     drivingConnections.append(poissonSourceAxon.PoissonAxon(tau, predictionELambda, drivingOnset, drivingOffset, offLambda))
#     neuron.addInput_a(drivingConnections[-1], weight)
for n in range(correlatedPredictionCount):
    weightE = predictionEWeight * gauss(1.0, 0.1)
    weightI = -1 * predictionIWeight * gauss(1.0, 0.1)
    predictionEConnections.append(poissonSourceAxon.PoissonAxon(tau, predictionELambda, predictionEOnset, sensoryEOffset, offLambda))
    neuron.addInput_b(predictionEConnections[-1], weightE)
    neuron.addInput_b(predictionEConnections[-1], weightI)

for n in range(unCorrelatedPredictionCount):
    weightE = predictionEWeight * gauss(1.0, 0.1)
    weightI = -1 * predictionIWeight * gauss(1.0, 0.1)
    predictionEConnections.append(poissonSourceAxon.PoissonAxon(tau, predictionELambda, predictionEOnset, sensoryEOffset, offLambda))
    neuron.addInput_b(predictionEConnections[-1], weightE)
    predictionIConnections.append(poissonSourceAxon.PoissonAxon(tau, predictionILambda, predictionIOnset, sensoryIOffset, offLambda))
    neuron.addInput_b(predictionIConnections[-1], weightI)

# Set up sensory inputs
for n in range(correlatedSensoryCount):
    weightE = sensoryEWeight * gauss(1.0, 0.1)
    weightI = -1 * sensoryIWeight * gauss(1.0, 0.1)
    sensoryEConnections.append(poissonSourceAxon.PoissonAxon(tau, sensoryELambda, sensoryEOnset, sensoryEOffset, offLambda))
    neuron.addInput_a(sensoryEConnections[-1], weightE)
    neuron.addInput_a(sensoryEConnections[-1], weightI)

for n in range(unCorrelatedSensoryCount):
    weightE = sensoryEWeight * gauss(1.0, 0.1)
    weightI = -1 * sensoryIWeight * gauss(1.0, 0.1)
    sensoryEConnections.append(poissonSourceAxon.PoissonAxon(tau, sensoryELambda, sensoryEOnset, sensoryEOffset, offLambda))
    neuron.addInput_a(sensoryEConnections[-1], weightE)
    sensoryIConnections.append(poissonSourceAxon.PoissonAxon(tau, sensoryILambda, sensoryIOnset, sensoryIOffset, offLambda))
    neuron.addInput_a(sensoryIConnections[-1], weightI)

# Set up records
KCNQOpen = zeros(int(maxTimeInMS/tau))
apexDV = zeros(int(maxTimeInMS/tau))
baseDV = zeros(int(maxTimeInMS/tau))
KCNQTau = zeros(int(maxTimeInMS/tau))
MCurrent = zeros(int(maxTimeInMS/tau))
Voltage = zeros(int(maxTimeInMS/tau))
hRecord = zeros(int(maxTimeInMS/tau))
electrodeTrace = zeros(int(maxTimeInMS/tau))
leak = zeros(int(maxTimeInMS/tau))
ampa = zeros(int(maxTimeInMS/tau))
gaba = zeros(int(maxTimeInMS/tau))
nmda = zeros(int(maxTimeInMS/tau))
apex_ampa = zeros(int(maxTimeInMS/tau))
apex_gaba = zeros(int(maxTimeInMS/tau))
apex_nmda = zeros(int(maxTimeInMS/tau))
external = zeros(int(maxTimeInMS/tau))
ampaPredictionRaster = zeros((predictionInputCount, int(maxTimeInMS/tau)))
nmdaPredictionRaster = zeros((predictionInputCount, int(maxTimeInMS/tau)))
gabaPredictionRaster = zeros((predictionInputCount, int(maxTimeInMS/tau)))
ampaSensoryRaster = zeros((sensoryInputCount, int(maxTimeInMS/tau)))
nmdaSensoryRaster = zeros((sensoryInputCount, int(maxTimeInMS/tau)))
gabaSensoryRaster = zeros((sensoryInputCount, int(maxTimeInMS/tau)))


# Run the simluation
for time in range(len(tspan)):
    # if time > drivingOnset:
    #     neuron.base.debug = True
    print("time:", time)
    neuron.step(time)
    KCNQOpen[time] = neuron.base.z
    KCNQTau[time] = neuron.base.tau_z
    MCurrent[time] = -neuron.base.gdapt*neuron.base.z*(neuron.base.v-neuron.base.v_K)
    Voltage[time] = neuron.base.v
    hRecord[time] = neuron.base.h
    apexDV[time] = neuron.apex.dv_temp
    baseDV[time] = neuron.base.dv_temp
    leak[time] = -neuron.base.g_shunt*(neuron.base.v-neuron.base.v_shunt)
    electrodeTrace[time] = (((neuron.base.IExternal - neuron.apex.IExternal) * neuronHeight * cos(0)) / (
            4 * pi * 8.854187817e-12 * (0.01) ** 2)) / 100000000000
    ampa[time] = neuron.base.I_ampa
    gaba[time] = neuron.base.I_gaba
    nmda[time] = neuron.base.I_nmda
    apex_ampa[time] = neuron.apex.I_ampa
    apex_gaba[time] = neuron.apex.I_gaba
    apex_nmda[time] = neuron.apex.I_nmda
    external[time] = neuron.base.IExternal

    # Snag raster data
    ampaPredictionRaster[:, time] = neuron.base.ampaRecord
    nmdaPredictionRaster[:, time] = neuron.base.nmdaRecord
    gabaPredictionRaster[:, time] = neuron.base.gabaRecord
    ampaSensoryRaster[:, time] = neuron.apex.ampaRecord
    nmdaSensoryRaster[:, time] = neuron.apex.nmdaRecord
    gabaSensoryRaster[:, time] = neuron.apex.gabaRecord

    for connection in predictionEConnections:
        connection.step()
    for connection in predictionIConnections:
        connection.step()
    for connection in sensoryEConnections:
        connection.step()
    for connection in sensoryIConnections:
        connection.step()

# Plot and save results
figDir = 'CA1Plots'

# Lowpass filter the simulated electrode measurements
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 10000.0       # sample rate, Hz
cutoff = 14  # desired cutoff frequency of the filter, Hz

# # Uncomment this to plot the frequency response of the filter we're using, if you are curious.
# b, a = butter_lowpass(cutoff, fs, order)
# w, h = freqz(b, a, worN=8000)
# plt.subplot(2, 1, 1)
# plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
# plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()

filteredElectrodeTracePrimed = butter_lowpass_filter(electrodeTrace, cutoff, fs, order)

# Uncomment this to plot the filtered electric field trace
figure()
plot(arange(0,maxTimeInMS,tau), filteredElectrodeTracePrimed, linewidth=1.5)
title('Electric Field: ' + primedTerm + ' Comparator')
xlabel('Time in ms')
ylabel('Voltage in mV')
a = gca()
ylim([-10,7])
savefig(figDir + '/' + pharmas + '_' +primedTerm+ 'ComparatorElectrodeTrace.png')
savefig(figDir + '/' + pharmas + '_' +primedTerm+ 'ComparatorElectrodeTrace.pdf')

# Uncomment this to plot the M-Current
figure()
plot(arange(0,maxTimeInMS,tau),MCurrent, linewidth=1.5)
title(primedTerm+' Comparator M-Current')
xlabel("Time in MS")
ylabel("Current in mA")
a = gca()
ylim([-400,20])
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ComparatorMCurrent.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ComparatorMCurrent.pdf')

# Uncomment this to plot the Basal Voltage Trace
figure()
plot(arange(0,maxTimeInMS,tau),Voltage, linewidth=1.5)
title(primedTerm + ' Comparator Voltage Trace')
xlabel("Time in MS")
ylabel("Voltage in mV")
a = gca()
ylim([-80,40])
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ComparatorVoltage.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ComparatorVoltage.pdf')

# Uncomment this to plot the basal AMPA current
figure()
plot(arange(0,maxTimeInMS,tau),ampa, linewidth=1.5)
title(primedTerm + ' basal AMPA')
xlabel("Time in MS")
ylabel("Current")
a = gca()
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'BasalAMPA.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'BasalAMPA.pdf')

# Uncomment this to plot the apical AMPA current
figure()
plot(arange(0,maxTimeInMS,tau), apex_ampa, linewidth=1.5)
title(primedTerm + ' apical AMPA')
xlabel("Time in MS")
ylabel("Current")
a = gca()
# ylim([-80,40])
a.set_xticklabels(a.get_xticks())
a.set_yticklabels(a.get_yticks())
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ApexAMPA.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ApexAMPA.pdf')

# Uncomment this to plot the basal GABA current
figure()
plot(arange(0,maxTimeInMS,tau),gaba, linewidth=1.5)
title(primedTerm+' Basal GABA')
xlabel("Time in MS")
ylabel("Current")
a = gca()
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'BasalGABA.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'BasalGABA.pdf')

# Uncomment this to plot the apical GABA current
figure()
plot(arange(0,maxTimeInMS,tau), apex_gaba, linewidth=1.5)
title(primedTerm+' Apical GABA')
xlabel("Time in MS")
ylabel("Current")
a = gca()
# ylim([-80,40])
a.set_xticklabels(a.get_xticks())
a.set_yticklabels(a.get_yticks())
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ApexGABA.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ApexGABA.pdf')

# Uncomment this to plot the basal NMDA current
figure()
plot(arange(0,maxTimeInMS,tau),nmda, linewidth=1.5)
title(primedTerm+' Basal NMDA')
xlabel("Time in MS")
ylabel("Current")
a = gca()
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'BasalNMDA.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'BasalNMDA.pdf')

# Uncomment this to plot the apical NMDA current
figure()
plot(arange(0,maxTimeInMS,tau),apex_nmda, linewidth=1.5)
title(primedTerm+' Apex NMDA')
xlabel("Time in MS")
ylabel("Current")
a = gca()
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ApexNMDA.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'ApexNMDA.pdf')

# Uncomment this to plot the axial current
figure()
plot(arange(0,maxTimeInMS,tau), external, linewidth=1.5)
title(primedTerm+' Axial Current')
xlabel("Time in MS")
ylabel("Current")
a = gca()
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'Axial.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'Axial.pdf')

# Uncomment this to generate a 3D plot of the basal voltage and M-Current, similar to Prescott et al. 2006
fig =figure()
ax = fig.gca(projection='3d')
ax.plot(arange(0,maxTimeInMS,tau), Voltage, MCurrent)
title(primedTerm+' Membrane Voltage and M-Current over Time')
xlabel("Time in ms")
ylabel("Membrane Voltage in mV")
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'VoltageMCurrent.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'VoltageMCurrent.pdf')

# Uncomment this to plot the basal KCNQ open proportion
figure()
plot(arange(0,maxTimeInMS,tau),KCNQOpen)
title(primedTerm+' proportion of KCNQ Channels Open')
xlabel("Time in MS")
ylabel("Proportion (1.0=100%)")
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'KCNQOpen.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'KCNQOpen.pdf')

# Uncomment this to plot the basal KCNQ time constant
figure()
plot(arange(0,maxTimeInMS,tau),KCNQTau)
title(primedTerm+' KCNQ Time Constant')
xlabel("Time in MS")
ylabel("Time Constant in MS")
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'KCNQTau.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'KCNQTau.pdf')

# Uncomment this to plot the basal Chloride leak current
figure()
plot(arange(0,maxTimeInMS,tau),leak)
title(primedTerm+' Leak Current')
xlabel("Time in MS")
ylabel("mA")
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'Leak.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'Leak.pdf')

# Uncomment this to plot the raster of input populations
predictionERaster = ampaPredictionRaster + nmdaPredictionRaster
sensoryERaster = ampaSensoryRaster + nmdaSensoryRaster
figure(figsize=(10,2))
pcolor(predictionERaster, cmap='Greys')
colorbar()
title(primedTerm + ' Prediction Excitatory Voltages Raster')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'PEVRaster.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'PEVRaster.pdf')

figure(figsize=(10,2))
pcolor(-1*gabaPredictionRaster, cmap='Greys')
colorbar()
title(primedTerm + ' Prediction Inhibitory Voltages Raster')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'PIVRaster.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'PIVRaster.pdf')

figure(figsize=(10,2))
pcolor(sensoryERaster, cmap='Greys')
colorbar()
title(primedTerm + ' Sensory Excitatory Voltages Raster')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'SEVRaster.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'SEVRaster.pdf')

figure(figsize=(10,2))
pcolor(-1*gabaSensoryRaster, cmap='Greys')
colorbar()
title(primedTerm + ' Sensory Inhibitory Voltages Raster')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'SIVRaster.png')
savefig(figDir + '/' + pharmas + '_' + primedTerm + 'SIVRaster.pdf')

show()