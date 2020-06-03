from __future__ import division

'''
Created on Sep 16, 2011

@author: stefan
'''

import monopoleNeuron
# import monopoleNeuronHillock
from scipy import *
from numpy import *

class DipoleWrapperNeuron:
    def __init__(self,inputs_a=[], inputs_b=[], outputs=[], externalInput_a = 0.0, externalInput_b = 0.0, position = [0.0, 0.0, 0.0], distance=0.06, debug=False, tau=0.1, tspan=range((10000)), inactivating=True):
        self.debug = debug
        self.tspan = tspan
        self.axialResistance = 250 #Not exactly straight compartmental axial resistance, since it covers all forms of transmission from apical to basal segments, including dendritic spikes.

        self.distance = distance
        if inputs_a is None:
            inputs_a = []
        if inputs_b is None:
            inputs_b = []
        if outputs is None:
            outputs = []
        apexPosition = position
        apexPosition[1] += distance

        self.apex = monopoleNeuron.MonopoleNeuron(inputs_a, outputs, externalInput_a, position, debug, tau, tspan=self.tspan, inactivating=inactivating)
        self.base = monopoleNeuron.MonopoleNeuron(inputs_b, outputs, externalInput_b, position, debug, tau, tspan=self.tspan, inactivating=inactivating)
        # self.hillock = monopoleNeuronHillock.MonopoleNeuronHillock(inputs_b, outputs, externalInput_b, position, debug, tau,
        #                                           tspan=self.tspan, inactivating=inactivating)

        # Time
        self.timestep = tau
        self.tau = tau
        self.position = position

    def step(self, time, externalInputA = 0, externalInputB = 0):
        externalInputH = 0

        # Calculate Bridge Current
        self.I_Bridge = (self.apex.v - self.base.v) / (self.axialResistance * self.distance)
        self.I_a = externalInputA - self.I_Bridge
        self.I_b = externalInputB + self.I_Bridge

        # # Calculate Bridge Current 2
        # self.I_Bridge2 = (self.base.v - self.hillock.v) / (self.axialResistance/10 * self.distance)
        # self.I_b2 = externalInputB - self.I_Bridge
        # self.I_h2 = externalInputH + self.I_Bridge2

        self.apex.step(time, self.I_a)
        self.base.step(time, self.I_b)
        # self.base.step(time, self.I_b + self.I_b2)
        # self.hillock.step(time, self.I_h2)

    def getState(self):
        vars = {}
        vars["v"] = self.base.v
        vars["i_a"] = self.I_a
        vars["i_b"] = self.I_b
        vars["I_Bridge"] = self.I_Bridge
        return vars 
    
    def addInput_a(self, tempInput, weight):
        self.apex.addInput(tempInput, weight)
        
    def addInput_b(self, tempInput, weight):
        self.base.addInput(tempInput, weight)
        
    def addOutput(self, output):
        self.base.addOutput(output)
        
    def getInputs_a(self):
        return self.apex.inputs
    
    def getInputs_b(self):
        return self.base.inputs

    def isBursting(self, inputOnset = 500):
        onsetSpikeRate = mean(self.base.spikeRecord[inputOnset:inputOnset+100])
        sustainedSpikeRate = mean(self.base.spikeRecord[inputOnset+101:-1])
        if onsetSpikeRate / sustainedSpikeRate > 3.0:
            return 1
        else:
            return 0