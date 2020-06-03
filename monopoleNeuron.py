from __future__ import division

'''
Created on Sep 16, 2011

@author: stefan
'''

from scipy import *
from numpy import *

class MonopoleNeuron:
    def __init__(self,inputs=None, outputs=None, externalInput = 0.0, position = [0.0, 0.0, 0.0], debug=False, tau=0.1, surfaceArea=1.0, tspan=range((10000)), inactivating=True):
        # print("Initialized!")
        
        self.debug = debug
        self.tspan = tspan
        self.inactivating = inactivating
        
        # Constants
        ## Original
        self.IExternal = 0
        self.C = 2 #2               #muF/cm^2
        self.g_Na = 20 #20           #mS/cm^2
        self.v_Na = 60          #mV
        self.g_K = 8  #8 #20            #mS/cm^2
        self.v_K = -100          #mV
        self.g_shunt = 2.1 #2.1 #1.2 #0.85      #mS/cm^2 (was varied from 2 to 5.3 in the paper)
        self.v_shunt = -70       #mV
        self.gdapt = 4        #mS/cm^2 (for m-current adaptation)
        self.gAHP = 1          #mS/cm^2
        self.vAHP = self.v_K
        self.alphaAHP = 0.005
        self.betaAHP = 0
        self.gammaAHP = 10 #5
        self.tauAHP = 200
        self.vdapt = self.v_K       #mV (for potassium m-current adaptation)
        self.beta = -35          #mV  (m-current threshold, seems different from what I've read in the past)      
        self.gamma = 10 #5           #mV
        self.alpha = 0.005       #unitless
        self.phi = 0.15 #0.15          #unitless
        self.v_1 = -1.2          #mV
        self.v_2 = 23 #23            #mV
        self.v_3 = -2 #-2            #mV
        self.v_4 = 21 #21            #mV
        if inputs is None:
            self.inputs = []
            self.weights = []
        else:
            self.inputs = inputs
            self.weights = [inputs[a].weight for a in self.inputs]
        if outputs is None:
            self.outputs = []
        else:
            self.outputs = outputs
        self.tau = tau
        self.position = position
        self.externalInput = externalInput
        self.surfaceArea = 1.0 #surfaceArea

        # Variables
        ## Dynamics
        self.v = -60
        self.w = 0
        self.z = 0.01
        self.ahp = 0.01
        self.m = 0

        ## Synaptic Parameters and variables
        self.I_syn = 0
        self.I_ampa = 0
        self.I_gaba = 0
        self.I_nmda = 0
        
        self.s_ampa = [0 for n in range(len(inputs))]
        self.s_gaba = [0 for n in range(len(inputs))]
        self.s_nmda = [0 for n in range(len(inputs))]
        self.s_nmda_x = [0 for n in range(len(inputs))]
        self.s_ampa_dt = [0 for n in range(len(inputs))]
        self.s_gaba_dt = [0 for n in range(len(inputs))]
        self.s_nmda_dt = [0 for n in range(len(inputs))]
        self.s_nmda_x_dt = [0 for n in range(len(inputs))]
        
        self.g_ampa = 7.5e-3
        self.g_gaba = 7.5e-3
        self.g_nmda = 2.0e-3#2e-3 # Try setting this to -4 instead, to cut down NMDA contribution
        self.mg = 1e-3
        self.tau_ampa = 2 #ms
        self.tau_gaba = 10 #10 #ms
        self.tau_nmda_rise = 2 #ms
        self.tau_nmda_decay = 100 #ms
        self.alpha2 = 0.5 # ms
        self.h = 1.0 #Percent Na channels open

        ##M-current related
        Cml = 1      # uF/cm2
        self.celsius = 36
        self.tau_m_peak = 1000 / 2.3**((self.celsius-36)/10)

        ## Instants
        self.na_proportion = 0.5 * (1 + tanh((self.v - self.v_1) / self.v_2))
        self.w_inf = 0.5 * (1 + tanh((self.v - self.v_3) / self.v_4))
        self.tau_w = 1 / (1 * cosh((self.v - self.v_3) / (2 * self.v_4)))
        self.tau_m = self.tau_m_peak / (3.3 * exp((self.v + 35) / 20) + exp(-(self.v + 35) / 20))

        self.naCurrent = self.surfaceArea * self.g_Na * self.na_proportion * (self.v - self.v_Na)
        self.kCurrent = self.surfaceArea * self.g_K * self.w * (self.v - self.v_K)
        self.shuntCurrent = self.surfaceArea * self.g_shunt * (self.v - self.v_shunt)
        self.mCurrent = self.surfaceArea * self.gdapt * self.z * (self.v - self.v_K)
        
        # Records
        self.vRecord = [-100 for f in range(len(self.tspan))]
        self.wRecord = [-100 for f in range(len(self.tspan))]
        self.zRecord = [-100 for f in range(len(self.tspan))]
        self.zTauRecord = [-100 for f in range(len(self.tspan))]
        self.mCurrentRecord = [-100 for f in range(len(self.tspan))]
        self.spikes = 0
        self.spikeRecord=[]
        self.currentlySpiking = False
        
    def stepDuctClamp(self, time, conductances):
        # Get inputs
        vars = {}
        
        # Determine Inputs
        self.currentExcitatoryInputs = []
        self.currentInhibitoryInputs = []
        self.I_syn = 0
        self.I_ampa = 0
        self.I_gaba = 0
        self.I_nmda = 0
        
    def step(self, time, externalCurrent = 0.0):
        # Get inputs
        vars = {}
        
        # Determine Inputs
        self.currentExcitatoryInputs = []
        self.currentInhibitoryInputs = []
        self.I_syn = 0
        self.I_ampa = 0
        self.I_gaba = 0
        self.I_nmda = 0
        
        if self.debug:
            print("Gaba by synapse:")
        for a in range(len(self.inputs)):
            axon = self.inputs[a]
            tempInput = axon.getInput(self)
            if tempInput > 0 and self.weights[a] > 0: # Glutamatergic input
                self.s_ampa_dt[a] = -(self.s_ampa[a] / self.tau_ampa) + 1
                self.s_gaba_dt[a] = -(self.s_gaba[a] / self.tau_gaba) 
                self.s_nmda_x_dt[a] = -(self.s_nmda_x[a] / self.tau_nmda_rise) + 1
                self.s_nmda_dt[a] = -(self.s_nmda[a] / self.tau_nmda_decay) + self.alpha2*self.s_nmda_x[a]*(1-self.s_nmda[a])
            elif tempInput > 0 and self.weights[a] < 0: #Gabaergic input 
                self.s_ampa_dt[a] = -self.s_ampa[a] / self.tau_ampa
                self.s_gaba_dt[a] = -(self.s_gaba[a] / self.tau_gaba) + 1
                self.s_nmda_x_dt[a] = -(self.s_nmda_x[a] / self.tau_nmda_rise)
                self.s_nmda_dt[a] = -(self.s_nmda[a] / self.tau_nmda_decay) + self.alpha2*self.s_nmda_x[a]*(1-self.s_nmda[a])
            else: # No input
                self.s_ampa_dt[a] = -self.s_ampa[a] / self.tau_ampa
                self.s_gaba_dt[a] = -self.s_gaba[a] / self.tau_gaba
                self.s_nmda_x_dt[a] = -(self.s_nmda_x[a] / self.tau_nmda_rise)
                self.s_nmda_dt[a] = -(self.s_nmda[a] / self.tau_nmda_decay) + self.alpha2*self.s_nmda_x[a]*(1-self.s_nmda[a])
            
            # Now update the actual value of the dynamic terms
            self.s_ampa[a] = self.s_ampa[a] + self.s_ampa_dt[a]*self.tau
            self.s_gaba[a] = self.s_gaba[a] + self.s_gaba_dt[a]*self.tau
            if self.debug:
                print("Synapse", a, ":", self.s_gaba[a])
            self.s_nmda[a] = self.s_nmda[a] + self.s_nmda_dt[a]*self.tau
            self.s_nmda_x[a] = self.s_nmda_x[a] + self.s_nmda_x_dt[a]*self.tau

        self.ampaRecord = [self.weights[a] * self.s_ampa[a] for a in range(len(self.inputs)) if self.weights[a] > 0]
        self.nmdaRecord = [self.weights[a] * self.s_nmda[a] for a in range(len(self.inputs)) if self.weights[a] > 0]
        self.gabaRecord = [self.weights[a] * self.s_gaba[a] for a in range(len(self.inputs)) if self.weights[a] < 0]
        self.I_ampa = sum([self.weights[a] * self.s_ampa[a] for a in range(len(self.inputs))])
        self.I_gaba = sum([-self.weights[a] * self.s_gaba[a] for a in range(len(self.inputs))])
        self.I_nmda = sum([self.weights[a] * self.s_nmda[a] for a in range(len(self.inputs))])
        self.ampaOpen = self.I_ampa
        self.gabaOpen = self.I_gaba
        self.nmdaOpen = self.I_nmda
        self.I_ampa = self.I_ampa * (self.g_ampa * (self.v - self.v_Na))
        self.I_gaba = self.I_gaba * (self.g_gaba * (self.v - self.v_shunt))
        self.I_nmda = self.I_nmda * ((self.g_nmda * (self.v - self.v_Na)) / 1 + self.mg * exp((-0.062*self.v) / 3.57))
        self.I_syn = self.I_ampa + self.I_gaba + self.I_nmda
        if self.debug:
            print("------------------")
            print("Voltage:", self.v)
            print("AMPA:", self.I_ampa)
            print("GABA:", self.I_gaba)
            print("NMDA:", self.I_nmda)
            print("External:", self.externalInput)
        Itotal = externalCurrent + self.externalInput - self.I_syn

        # Update Instants
        # self.na_proportion_inf = 0.5 * (1 + tanh((self.v - self.v_1) / self.v_2))
        self.na_proportion = 0.5 * (1 + tanh((self.v - self.v_1) / self.v_2))
        self.tauh = ((2 * 232 / pi) * (28 / (4 * (
                    self.v + 64) ** 2 + 28 ** 2)))  # Equivalent to the equation in Fernandez, Mehaffey, & Turner, 2005
        self.hinf = (1 / (1 + exp(-(self.v + 40) / -3)))

        if self.debug:
            print("wInf:", 0.5*(1+tanh((self.v-self.v_3)/self.v_4)))
        self.w_inf = 0.5*(1+tanh((self.v-self.v_3)/self.v_4))
        self.tau_w = 1/cosh((self.v-self.v_3)/(2*self.v_4)) #was 2x
        self.h = (self.hinf - ((self.hinf - self.h) * (exp(-(self.tau) / (self.tauh)))))
        self.tau_z_peak = 1000 / 2.3 ** ((self.celsius - 36) / 10)
        self.tau_z = self.tau_z_peak / (3.3 * exp((self.v - self.beta) / 20) + exp(-(self.v - self.beta) / 20))
    
        # Update Dynamics
        # self.na_proportion = self.na_proportion + (self.tau * (self.na_proportion_inf - self.na_proportion) / 0.5)
        self.w = self.w + self.tau * (self.phi * (self.w_inf - self.w) / (self.tau_w))
        self.z = self.z + self.tau * ((1/(1+exp((self.beta-self.v)/self.gamma))-self.z) / (self.tau_z))
        # self.ahp = self.ahp + self.tau * self.alphaAHP * (1/(1+exp((self.betaAHP-self.v)/self.gammaAHP))-self.ahp) # From Prescott et al. 2006
        
        if self.debug:
            print("Itotal:", Itotal)
            # print("Na Passive:", -self.g_Na*self.m_inf*(self.v-self.v_Na))
            print("W:", self.w)
            print("K Voltage Diff:", (self.v-self.v_K))
            print("K Passive:", -self.g_K*self.w*(self.v-self.v_K))
            print("Shunting Passive:", -self.g_shunt*(self.v-self.v_shunt))
            print("M-Current:", -self.gdapt*self.z*(self.v-self.v_K))
                 
        if self.inactivating == True:
            activeNA = 1 - self.h
        else:
            activeNA = 1

        self.naCurrent = self.surfaceArea * self.g_Na * self.na_proportion * activeNA * (self.v - self.v_Na)
        self.kCurrent = self.surfaceArea * self.g_K * self.w * (self.v - self.v_K)
        self.shuntCurrent = self.surfaceArea * self.g_shunt * (self.v - self.v_shunt)
        self.mCurrent = self.surfaceArea * self.gdapt * self.z * (self.v - self.v_K)
        self.IExternal = (self.naCurrent - self.kCurrent - self.shuntCurrent - self.mCurrent) / (
                    self.C * (1 / self.tau))
        self.dv_temp = (Itotal - self.naCurrent - self.kCurrent - self.shuntCurrent - self.mCurrent) / (
                    self.C * (1 / self.tau))
        self.v = self.v + self.dv_temp
        
        # Record Info
        # self.vRecord.append(self.v)
        # self.wRecord.append(self.w)
        # self.zRecord.append(self.z)
        # self.zTauRecord.append(self.tau_z)
        # self.mCurrentRecord.append(-self.gdapt*self.z*(self.v-self.v_K))
        if self.currentlySpiking == False and self.v > 0:
            self.spikes += 1
            self.spikeRecord.append([time,1])
            self.currentlySpiking = True
            for output in self.outputs:
                output.enqueue()
        if self.currentlySpiking == True and self.v < -10:
            self.currentlySpiking = False        

    def getState(self):
        vars = {}
        vars["v"] = self.v
        return vars
    
    def addInput(self, incomingAxon, weight):
        incomingAxon.addTarget(self)
        self.inputs.append(incomingAxon)
        self.weights.append(weight)
        self.s_ampa.append(0)
        self.s_gaba.append(0)
        self.s_nmda.append(0)
        self.s_nmda_x.append(0)
        self.s_ampa_dt.append(0)
        self.s_gaba_dt.append(0)
        self.s_nmda_dt.append(0)
        self.s_nmda_x_dt.append(0)
        
    def addOutput(self, output):
        self.outputs.append(output)
        
    def getInputs(self):
        return self.inputs
