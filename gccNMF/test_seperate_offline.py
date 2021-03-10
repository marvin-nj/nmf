from gccNMFFunctions import *
from gccNMFPlotting import *
from IPython import display

#2021.03.10  by marvin  
# need two channels  data
#

# Preprocessing params
windowSize = 1024
fftSize = windowSize
hopSize = 128
windowFunction = hanning

# TDOA params
numTDOAs = 128

# NMF params
dictionarySize = 128
numIterations = 100
sparsityAlpha = 0

# Input params    
mixtureFileNamePrefix = 'data/mix_8k_1-8-noSil'
microphoneSeparationInMetres = 1.0
numSources = 2


mixtureFileName = getMixtureFileName(mixtureFileNamePrefix)
stereoSamples, sampleRate = loadMixtureSignal(mixtureFileName)
stereoSamples=stereoSamples.reshape((1,-1))
numChannels,numSamples = stereoSamples.shape
durationInSeconds = numSamples / float(sampleRate)

#describeMixtureSignal(stereoSamples, sampleRate)
#figure(figsize=(14, 6))
#plotMixtureSignal(stereoSamples, sampleRate)
#display.display( display.Audio(mixtureFileName) )

complexMixtureSpectrogram = computeComplexMixtureSpectrogram( stereoSamples, windowSize,
                                                              hopSize, windowFunction ) 

numChannels,numFrequencies, numTime = complexMixtureSpectrogram.shape
print(numChannels,numFrequencies, numTime)
frequenciesInHz = getFrequenciesInHz(sampleRate, numFrequencies)
frequenciesInkHz = frequenciesInHz / 1000.0

#describeMixtureSpectrograms(windowSize, hopSize, windowFunction, complexMixtureSpectrogram)
#figure(figsize=(12, 8))
#plotMixtureSpectrograms(complexMixtureSpectrogram, frequenciesInkHz, durationInSeconds)

#spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() \
#                     / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])
spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[0].conj() \
                     / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[0])

angularSpectrogram = getAngularSpectrogram( spectralCoherenceV, frequenciesInHz,
                                            microphoneSeparationInMetres, numTDOAs )
meanAngularSpectrum = mean(angularSpectrogram, axis=-1) 
targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum( meanAngularSpectrum,
                                                                  microphoneSeparationInMetres,
                                                                  numTDOAs, numSources)

#figure(figsize=(14, 6))
#plotGCCPHATLocalization( spectralCoherenceV, angularSpectrogram, meanAngularSpectrum,targetTDOAIndexes, microphoneSeparationInMetres, numTDOAs,durationInSeconds )

V = concatenate( abs(complexMixtureSpectrogram), axis=-1 )
W, H = performKLNMF(V, dictionarySize, numIterations, sparsityAlpha)

numChannels = stereoSamples.shape[0]
stereoH = array( hsplit(H, numChannels) )

#describeNMFDecomposition(V, W, H)
#figure(figsize=(12, 12))
#plotNMFDecomposition(V, W, H, frequenciesInkHz, durationInSeconds, numAtomsToPlot=15)

targetTDOAGCCNMFs = getTargetTDOAGCCNMFs( spectralCoherenceV, microphoneSeparationInMetres,
                                          numTDOAs, frequenciesInHz, targetTDOAIndexes, W,
                                          stereoH )
targetCoefficientMasks = getTargetCoefficientMasks(targetTDOAGCCNMFs, numSources)

#figure(figsize=(12, 12))
#plotCoefficientMasks(targetCoefficientMasks, stereoH, durationInSeconds)

targetSpectrogramEstimates = getTargetSpectrogramEstimates( targetCoefficientMasks,
                                                            complexMixtureSpectrogram, W,
                                                            stereoH )

#figure(figsize=(12, 12))
#plotTargetSpectrogramEstimates(targetSpectrogramEstimates, durationInSeconds, frequenciesInkHz)

targetSignalEstimates = getTargetSignalEstimates( targetSpectrogramEstimates, windowSize,
                                                  hopSize, windowFunction )
saveTargetSignalEstimates(targetSignalEstimates, sampleRate, mixtureFileNamePrefix)

'''
for sourceIndex in range(numSources):
    figure(figsize=(14, 2))
    fileName = getSourceEstimateFileName(mixtureFileNamePrefix, sourceIndex)
    plotTargetSignalEstimate( targetSignalEstimates[sourceIndex], sampleRate,
                              'Source %d' % (sourceIndex+1) )
    display.display(display.Audio(fileName))
'''
