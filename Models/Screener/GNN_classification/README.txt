## Screening using multi labels and graphs

# Prosodic features
mean F0
standard deviation F0 
Median F0 

-> research shows that children with DLD have difficulties with prosodic and rhythmic treatment of speech, 
with reduced sensitivity to envelope amplitude variations. 
Prosody is linked to lexicon and grammar in children with DLD and these prosodic deficits can impact their language functioning. 
So I use Parselmouth (Python interface for Praat) to extract F0 as it is the reference tool in clinical phonetics.

# Intensity (vocal energy)
mean intensity
standard deviation intensity 

-> Variations in intensity reflect the respiratory control and laryngeal coordination which are often impaired in DLD.

# Temporal features
Speech/pause analysis
mean speech duration 
mean pause duration 
speech rate
speech to pause ratio 

Methodology: automatic detection of speech segments vs silence via energy thresholding 
25ms frames with 10ms hop (standard in speech analysis)
30th percentile threshold for robustness to intensity variations 

-> temporal patterns (pauses, rate) are established clinical markers of DLD, reflecting difficulties with language planning

# Spectral features
MFCC (mel frequency spectral coefficients)
mfcc 0 mean 
mfcc 0 standard deviation
…
mfcc 12 mean 
mfcc 12 standard deviation 

-> MFCCs capture spectral properties of speech related to articulation and vocal resonance. 13 coefficients = standard in speech recognition 

Other spectral features: spectral centroid, spectral bandwidth, spectral roll off, zero crossing rate 
-> These metrics quantify spectral brightness and frequency distribution, reflecting articulatory quality

# Vocal characteristics quality 
Jitter and Shimmer
local jitter
jitter rap 
shimmer local 
shimmer apq3 

-> Jitter measures cycle to cycle period variations and shimmer measures amplitude variations, caused by irregular vibrations of the vocal folds. 
                      They are perceived as roughness or hoarseness.
                      

Implementation 
local jitter: average variation over 3 periods
local shimmer: cycle to cycle amplitude variation 
APQ3 shimmer: 3-point amplitude disturbance quotient 

# HNR (Harmonics to Noise ratio) 
hnr mean 
hnr standard deviation 

-> HNR is a key parameter in acoustic analysis for assessing voice quality showing significant differences between normal subjects and patients with voice disorders.


## Graph structure 

# Prosodic relations: 
mean f0: std F0, range F0, intensity mean 
variations in F0 are linked together and with intensity by laryngeal control.

# Vocal quality relations: 
jitter local: jitter rap, shimmer local 
shimmer local: shimmer apq3, hnr mean 
jitter is affected by lack of control in vocal fold vibrations, and shimmer by reduced glottal resistance 
                      
# Cross domain relations
speech rate: spectral centroid mean, std F0 
mean pause duration: jitter local, shimmer local 
Hypothesis: in DLD, difficulties in planification (long pauses) can co occur with vocal instability (high jitter/ shimmer) 

# GNN Architecture
GNN can propagate information between connected nodes 
Normalisation BatchNorm can stabilize training with heterogeneous features 
Global pooling can aggregate graph information in fixed representation 

Each node represents one characteristic with its value as a node feature. 
