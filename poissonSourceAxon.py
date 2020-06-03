from __future__ import division
from numpy import *
from scipy.stats import poisson

'''
Created on Jan 8, 2011

An axon class that provides external input in the form of a poisson distribution of spikes.  It still maxes out at one spike per time point, however, so we need a number of them to get a serious distribution.

@author: stefan
'''

class PoissonAxon:
    def __init__(self, timeStep, lambdaConst = 0.5, onsetTime = 0, offsetTime = 10000000, offLambda = 0.001):
        self.lambdaConst = lambdaConst
        self.timeStep = timeStep
        self.time = 0
        self.onsetTime = onsetTime
        self.offsetTime = offsetTime
        self.distributionOn = poisson(self.lambdaConst)
        self.distributionOff = poisson(offLambda)
        self.output = 0
        self.vv = []
        self.rampEnd = self.onsetTime + 1 / self.timeStep

    def getInput(self, target):
        return self.output

    def addTarget(self, target):
        return True

    def step(self): 
        self.time += self.timeStep
        if self.time < self.onsetTime or self.time > self.offsetTime:
            self.output = self.distributionOff.rvs((1,))[0]
        else:
            # if self.time <= self.rampEnd:
            #     rampSteps = self.rampEnd - self.onsetTime
            #     rampStepped = self.time - self.onsetTime
            #     tempLambda = self.lambdaConst * rampStepped / rampSteps
            #     # print("TempLambda: ", tempLambda)
            #     self.distributionOn = poisson(tempLambda)
            self.output = self.distributionOn.rvs((1,))[0]
        self.vv.append(self.output)

