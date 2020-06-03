from __future__ import division
import dipoleWrapperNeuron
import poissonSourceAxon

from pylab import *
from random import randint, gauss
from numpy import *

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
T1=len(tspan)/5
T2 = 4*T1

# Neuron Parameters
neuronHeight = 0.00056
predictionEOnset = 300
predictionIOnset = 320
predictionEOffset = 1800
predictionIOffset = 1820
sensoryEOnset = 100
sensoryEOffset = 1800
sensoryIOnset = 120
sensoryIOffset = 1820
predictionEWeight = 15
predictionIWeight = 12.5
sensoryEWeight = 4
sensoryIWeight = 18
predictionELambda = 0.01
predictionILambda = 0.01
sensoryELambda = 0.01
sensoryILambda = 0.01
offLambda = 0.0001
predictionInputCount = 200
sensoryInputCount = 200
ratioOfCorrelatedInhibition = 0.5

correlatedPredictionCount = int(sensoryInputCount * ratioOfCorrelatedInhibition)
unCorrelatedPredictionCount = sensoryInputCount - correlatedPredictionCount

correlatedSensoryCount = int(predictionInputCount * ratioOfCorrelatedInhibition)
unCorrelatedSensoryCount = predictionInputCount - correlatedSensoryCount

# Set the pharmacological manipulations
ACh = False
Retigabine = False
RetigabinePartial = False
RetigabineRescue = False
PCP = False
PCPPartial = False

# Uncomment the label you want for the saved plots, indicating which pharmacological manipulations were present.
pharmas = 'Baseline'
# pharmas = 'PCP_Partial'
# pharmas = 'PCP'
# pharmas = 'Retigabine'
# pharmas = 'RetigabinePartial'
# pharmas = 'PCPPartialPlusRetigabine'
# pharmas = 'PCPPlusRetigabine' # Used this one.
# pharmas = 'PCPPlusRetigabinePartial'
# pharmas = 'PCPPlusRetigabineRescueWeight'
# pharmas = 'TargetPrimedBaseline'
# pharmas = 'ACh'
# pharmas = 'AChPerSE'
# pharmas = 'noEPrime'
# pharmas = 'noIPrime'

trialCount = 10
correlationMax = 1
correlationRange = linspace(1.0, 0, 11)
correlationRangeLen = len(correlationRange)
trialRec = 0
noiseIdxRec = 0

## Robustness records
results = []
TPRecord = zeros((trialCount, correlationRangeLen))
FPRecord = zeros((trialCount, correlationRangeLen))
FNRecord = zeros((trialCount, correlationRangeLen))
TNRecord = zeros((trialCount, correlationRangeLen))

# # Uncomment to resume where you left off
# pickleJar = open("pickle.jar", "rb")
# [trialRec, noiseIdxRec, FPRecord, TPRecord, TNRecord, FNRecord] = pickle.load(pickleJar) # primedVoltageRaster, unprimedVoltageRaster

for trial in range(trialRec, trialCount):
    print("TRIAL: ", trial)
    # Loop
    for noiseIdx in range(noiseIdxRec, correlationRangeLen):
        ratioOfCorrelatedInhibition = correlationRange[noiseIdx]
        print("\tCorrelated Proportion: ", ratioOfCorrelatedInhibition)
        correlatedPredictionCount = int(predictionInputCount * ratioOfCorrelatedInhibition)
        unCorrelatedPredictionCount = predictionInputCount - correlatedPredictionCount
        for primeLambdas in [0.01, 0.001]:
            sensoryELambda = primeLambdas
            sensoryILambda = primeLambdas

            # Set up the neuron
            neuron = dipoleWrapperNeuron.DipoleWrapperNeuron(inputs_a=[], inputs_b=[], outputs=[], externalInput_a=0.0, externalInput_b=0.0, distance = neuronHeight, position = [randint(1,100), randint(1,100), 0.0], debug=False, tau=tau, tspan=tspan, inactivating=False)

            if ACh:
                neuron.base.gdapt = 3.2
                neuron.apex.gdapt = 3.2

            if PCP:
                neuron.base.g_nmda = 1.75e-3
                neuron.apex.g_nmda = 1.75e-3
                sensoryILambda = 0.001
                predictionILambda = 0.001

            if PCPPartial:
                neuron.base.g_nmda = 1.9e-3
                neuron.apex.g_nmda = 1.9e-3
                sensoryILambda = 0.009

            if Retigabine:
                neuron.base.gdapt = 5.0
                neuron.apex.gdapt = 5.0
                sensoryIWeight += 6.0

            if RetigabinePartial:
                neuron.base.gdapt = 4.5
                neuron.apex.gdapt = 4.5
                sensoryIWeight += 5.5

            if RetigabineRescue:
                neuron.base.gdapt = 4.1
                neuron.apex.gdapt = 4.1
                sensoryIWeight += 4

            predictionEConnections = []
            predictionIConnections = []
            sensoryEConnections = []
            sensoryIConnections = []

            # Set up prediction inputs from CA3
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

            # Set up sensory inputs from EC3
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
            Voltage = zeros(int(maxTimeInMS / tau))

            # Run the simluation
            for time in range(len(tspan)):
                # if time > drivingOnset:
                #     neuron.base.debug = True
                # print("time:", time)
                neuron.step(time)
                Voltage[time] = neuron.base.v

                for connection in predictionEConnections:
                    connection.step()
                for connection in predictionIConnections:
                    connection.step()
                for connection in sensoryEConnections:
                    connection.step()
                for connection in sensoryIConnections:
                    connection.step()

            # Compute TP/TN/FP/FN
            onsetSpikeRate = len([neuron.base.spikeRecord[s] for s in range(len(neuron.base.spikeRecord)) if
                                  neuron.base.spikeRecord[s][0] > int(floor(predictionEOnset / tau)) and
                                  neuron.base.spikeRecord[s][0] < int(floor(predictionEOnset / tau + 200 / tau))])
            sustainedSpikeRate = len([neuron.base.spikeRecord[s] for s in range(len(neuron.base.spikeRecord)) if
                                      neuron.base.spikeRecord[s][0] > int(floor(predictionEOnset / tau + 201 / tau)) and
                                      neuron.base.spikeRecord[s][0] < int(floor(predictionEOffset / tau))])

            sum(neuron.base.spikeRecord[int(floor(predictionEOnset / tau + 201 / tau)): int(
                floor(predictionEOffset / tau))])

            # print(onsetSpikeRate)
            # print(sustainedSpikeRate)

            if ((onsetSpikeRate + 0.0001) / (sustainedSpikeRate + 0.0001) >= 2.0) and (sustainedSpikeRate < 2):
                phasic = 1
                # print "PHASIC!"
            else:
                phasic = 0

            if primeLambdas < 0.01 and phasic == 1:
                FPRecord[trial, noiseIdx] = 1
                print("False Positive!")
                figure()
                plot(arange(0, maxTimeInMS, tau), Voltage, linewidth=1.5)
                # plot(arange(0, maxTimeInMS, tau), noiseRecord, linewidth=1.0)
                title("False Positive with Noise:" + str(ratioOfCorrelatedInhibition))
                xlabel("Time in MS")
                ylabel("Voltage in mV")
                a = gca()
                ylim([-80, 40])
                # show()
            elif primeLambdas >= 0.01 and phasic == 1:
                TPRecord[trial, noiseIdx] = 1
                print("True Positive!")
            elif primeLambdas < 0.01 and phasic == 0:
                TNRecord[trial, noiseIdx] = 1
                print("True Negative!")
            elif primeLambdas >= 0.01 and phasic == 0:
                FNRecord[trial, noiseIdx] = 1
                print("False Negative!")
                figure()
                plot(arange(0, maxTimeInMS, tau), Voltage, linewidth=1.5)
                # plot(arange(0, maxTimeInMS, tau), noiseRecord, linewidth=1.0)
                title("False Negative with Noise:" + str(ratioOfCorrelatedInhibition))
                xlabel("Time in MS")
                ylabel("Voltage in mV")
                a = gca()
                ylim([-80, 40])
                # show()
            with open("pickle.jar", "wb") as pickleJar:
                pickle.dump([trial, noiseIdx, FPRecord, TPRecord, TNRecord, FNRecord], pickleJar)
    noiseIdxRec = 0

FPToPlot = mean(FPRecord, 0) / 2
TPToPlot = mean(TPRecord, 0) / 2
TNToPlot = mean(TNRecord, 0) / 2
FNToPlot = mean(FNRecord, 0) / 2

FPToPlot = [sum(FPRecord[:, n]) for n in range(correlationRangeLen)]
TPToPlot = [sum(TPRecord[:, n]) for n in range(correlationRangeLen)]
TNToPlot = [sum(TNRecord[:, n]) for n in range(correlationRangeLen)]
FNToPlot = [sum(FNRecord[:, n]) for n in range(correlationRangeLen)]

correlationRangeToPlot = linspace(10,0,11)

figure()
plot(correlationRangeToPlot, FPToPlot, linewidth=1.5, label="False Positives")
plot(correlationRangeToPlot, TPToPlot, linewidth=1.5, label="True Positives")
plot(correlationRangeToPlot, TNToPlot, linewidth=1.5, label="True Negatives")
plot(correlationRangeToPlot, FNToPlot, linewidth=1.5, label="False Negatives")
legend()
title("Performance with Decorrelated Inputs")
xlabel("Proportion of Linked E/I Inputs")
ylabel("Number of Trials")
a = gca()
# ylim([-0.1,1.1])
labels = [item.get_text() for item in a.get_xticklabels()]
labels[1] = 1.0
labels[2] = 0.8
labels[3] = 0.6
labels[4] = 0.4
labels[5] = 0.2
labels[6] = 0.0
a.set_xticklabels(labels)

figure()
plot(correlationRangeToPlot, FPToPlot, linewidth=1.5, label="False Positives")
plot(correlationRangeToPlot, TNToPlot, linewidth=1.5, label="True Negatives")
legend()
title("Performance with Decorrelated Inputs: Unprimed Trials")
xlabel("Proportion of Linked E/I Inputs")
ylabel("Number of Trials")
a = gca()
# ylim([-0.1,1.1])
labels = [item.get_text() for item in a.get_xticklabels()]
labels[1] = 1.0
labels[2] = 0.8
labels[3] = 0.6
labels[4] = 0.4
labels[5] = 0.2
labels[6] = 0.0
a.set_xticklabels(labels)

figure()
plot(correlationRangeToPlot, TPToPlot, linewidth=1.5, label="True Positives")
plot(correlationRangeToPlot, FNToPlot, linewidth=1.5, label="False Negatives")
legend()
title("Performance with Decorrelated Inputs: Primed Trials")
xlabel("Proportion of Linked E/I Inputs")
ylabel("Number of Trials")
a = gca()
# ylim([-0.1,1.1])
labels = [item.get_text() for item in a.get_xticklabels()]
labels[1] = 1.0
labels[2] = 0.8
labels[3] = 0.6
labels[4] = 0.4
labels[5] = 0.2
labels[6] = 0.0
a.set_xticklabels(labels)

show()
