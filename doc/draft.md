# Building an Indonesian Language Online Speaker Diarization System

# Abstract

Online speaker diarization is the process of determining 'who spoke when'
given an ongoing conversation or audio stream, in contrast to the offline
scenario where the conversation has concluded and the entire file is available.
The process of constructing an Indonesian language online speaker diarization
system is explored, from corpus development to software design and testing.
The system conducts speaker identification directly on silence separated
segments and employs a simple averaging post-processing method to boost
accuracy, resulting in a system with a one utterance latency for predictions.
Experimentation against a standard baseline offline system resulted in a
speaker error rate (SER) of 25.5% and *tbd* for the proposed and baseline
systems, respectively.

# 1 Background

Speaker diarization is the process of determining 'who spoke when', for
instance in a conference or interview setting. The task of speaker diarization
is to take audio containing speech from one or more speakers and determine the
identity of the speaker at any given time. In online speaker diarization, there
is the additional constraint of producing these identities in a timely manner,
be it in regular time intervals or at certain speech boundaries. Online speaker
diarization differs from offline speaker diarizationn mainly in the amount of
information available for analysis; online diarization makes use of all data
available up to the current time, whereas for offline diarization data at all
time points is available.

In general, the diarization process can be broken down into a number of steps.
In speaker segmentation, the incoming speech is split into segments containing
speech from a single speaker. In speaker clustering, the segments belonging to
the same speaker are grouped together. Finally, these groups of segments are
used to run speaker identification. Additional post-processing methods are used
to improve predictions, but are often dependent on analysis over the entire
length of speech, which may be unavailable in advance in the online scenario.

Online speaker diarization is important when real time or close to real time
information regarding speakers is required. Paired with online speech
recognition, for example, it becomes possible to generate a transcription of an
ongoing conference or meeting. Other uses include the addition of real time or
close to real time machine translation and summarization. This research
describes the process of constructing an online speaker diarization system,
detailing the corpus development, system design, and experiments.

Many approaches have been taken over

Diarization error rate (DER) is calculated as the per frame

Each meeting was subsequently analyzed to determine when speaker changes
occurred, known variously as speaker change detection, speaker turn
identification, and speaker segmentation. In an offline setting, this typically
consists of the following steps in most speech diarization systems:

1. Divide audio into frames.
2. First pass using BL.
3. Second pass using KL2.

In practice, speaker segmentation mostly succeeds in guaranteeing individual
segments contain at most one speaker, but fails to ensure consecutive segments
belong to different speakers.

This process is typically followed by hierarchical clustering of speakers
based on information dervied from the resulting speaker segments. In an offline
setting, this is achieved via the following steps:

1. Train a GMM for each speaker segment
2. Specify a difference metric as the basis for clustering
3. Cluster segments hierarchically using difference threshold as reference


# 2 Online Windowed Speaker Recognition

## 2.1 Related research


# 3 Corpus Development

## 3.1 Corpus Background

A speaker identification corpus was built for the purpose of this research,
with data taken from audio recordings of the meetings of the Indonesian
Regional Representation Council (Dewan Perwakilan Daerah Republik Indonesia)
throughout 2014 and 2015. These recordings were provided by PT Inti, an
Indonesian telecommunication state company, with whom we have jointly developed
portions of the Perisalah speech recognition system *ref*. A total of 57
separate meetings were recorded during this time period, 30 of which were
chosen for this research.

These meetings, which are open to the public, are held by members of the
Regional Representation Council (DPD) to discuss various national concerns
related to regional representation in the passing of new laws. The committee
primarily serves a legislative function, including monitoring and budgeting,
specifically in relation to regional aspirations. In addition to their primary
function, the council is also occasionally called upon to resolve conflicts
between regional stakeholders and the government. The subject matter and
structure of the meetings can generally be classified into one of the
following topics:

1. Expert consultation sessions. Experts in relevant fields are called upon to
assess the current status of the issue being discussed and are asked to present
their topic of expertise. These meetings will typically consist of one or more
experts presenting their topics uninterrupted, before concluding with a
question and answering session.
2. Discussing recommendations. These meetings are typically heavily moderated
discussion sessions, whereby the moderator goes through the prepared document
in sections and calls upon the relevant teams for clarification, before opening
discussion to the floor for that section. Although still relatively structured
due to the moderation, these meetings can at times produce portions of
overlapping speech.
3. Summons. Government officers are summoned to discuss issues related to their
performance or regarding matters of importance.
4. Conflict resolution. This involves a hearing between two sides in a conflict
usually regarding settlements related to land rights. The two sides are called
upon to make statements regarding the issue before the council, after which
the council will discuss the issue.
5. Internal sessions. These are usually meetings to discuss internal matters,
for instance with regards to scheduling future meetings, deadlines, diplomatic
visits to various regions, et cetera.

An overview of the meetings is presented in *table*.

Meeting ID        Topic                         Type
---------------   ----------------------------  ------------
2,3               Mother and child healthcare   Consultation
4,10,32,46        Scheduling                    Internal
5                 Tax ministry                  Summon
6,34,48           Tax law revisions             Discussion
7                 Bank Indonesia governor       Summon
8                 National auditing report      Discussion
9,18              Budget office                 Consultation
11,12,13          Budget office                 Discussion
21,29,30,49,50    Koperasi                      Consultation
38                Koperasi                      Discussion
33,40             Budget recommendations        Discussion
44                Land dispute                  Conflict
45,127            Agriculture ministry          Summon
53,57             National planning             Discussion

## 3.2 Pre-processing

The meetings were recorded using the Perisalah system, which combines the audio
from numerous microphones into a single channel and stores them for further
processing. These recordings were first converted to 16 kHz 16-bit WAV audio
for ease of storage and access. In the process, the audio was high pass
filtered at 100 Hz to reduce low frequency noises such as rumbling and pops.
In addition, loudness leveling was applied across recordings to ensure equal
average loudness between meetings and to prevent clipping of the signal. In
particular, loudness leveling is important if signal level/energy is an audio
feature. The audio recordings were then analyzed to determine the total
duration of speech they contained, as meetings were often preceded, succeeded,
and occasionally punctured by long stretches of silence.

## 3.3 Annotation

The annotation of training data was conducted in a semi-supervised and
iterative manner. This was due mainly to the difficulty of identifying speakers
across meetings, and also to reduce the time-costly manual labeling of speaker
segment boundaries. The LIUM speaker diarization toolkit was utilized for most
of the training, specifically for speaker modeling and identification and the
initial speaker clustering and labeling. Various Python scripts were also
utilized to ease the task of annotation and data analysis.

A single speaker clustering run was conducted on the pre-processed audio to
obtain the initial predictions for speaker segment boundaries and speaker
labels. Each segment in each meeting was then labeled manually to obtain
speaker labels and genders. For the purposes of this research, speaker segment
boundaries were not altered. Instead, segments with overlapping speakers were
marked and omitted from training and testing, and hence do not contribute to
the final error rate. As such, it is assumed that the segment boundaries
derived from the speaker segmentations process was accurate.

In the first labeling iteration, speaker labels were attached to each segment
by listening through the recordings and manually assigning a label. Due to
human limitations, it was difficult to ensure speaker labels were consistent
across files/meetings. Instead, an iterative approach was taken where:

1. Each meeting is labeled locally, with speaker labels that apply only for
the given file.
2. Each (local) speaker label is assumed to belong to a unique speaker across
all meetings and is automatically assigned a unique label.
3. A speaker model is trained from a given amount of speech from each unique
speaker.
4. The speaker model is cross-verified and a prediction produced.
5. Manually cross-check reference for incorrect speaker hypotheses; for
reference speaker labels with multiple hypothesized speaker labels, check if
the multiple hypothesized labels actually belong to the same speaker.
6. Manually assign a unique speaker label for such speakers, and modify the
reference to reflect this unique label. Hence, this step is an attempt to
label speakers consistently across meetings.
7. Repeat from step 2, but do not automatically assign a label to speakers
manually edited in step 6.

In this way, the reference was iteratively improved until mislabeled
speakers were no longer encountered. Figure 1 illustrates this process, with
data elements and processes depicted in rectangles and ellipses, respectively.

![Data flow diagram for experiment baseline](dfd_baseline.png)

In total, 236 speakers were identified. Due to the cross-checking method above,
though, speakers with a lack of training data could not be verified, and there
may be a number of double-counted speakers. The corpus contains 42.81 hours of
speech, discounting silence, noise, and overlapping speaker segments. On
average, each speaker has around 653.1 seconds of speech. However, due to the
moderated nature of the meetings a small number of speakers contributed a large
part of the recorded speech.

It may be useful to


# 4 Experiment and Results

## 4.1 Overview and Speaker Model

Experiments were conducted to ascertain the difference in speaker prediction
accuracy between the online speaker diarization system and a baseline offline
system. Both systems utilized the speaker model obtained in Section 3.3. The
online system utilizes a different speaker segmentation method, essentially
dividing the input audio stream at quiet sections and running speaker
identification directly on these short segments whilst forgoing the clustering
step. The baseline system utilizes the standard setup and is discussed below.
Accuracy evaluation was conducted on a per frame basis as speaker segment
boundaries and positions differed between the two methods.

The speaker model was built using various amounts of training data from the
cleaned corpus, with baseline testing conducted on the remainder of the data.
It is desirable to use the minimum amount of speech per speaker in training the
speaker model, as this correlates to a shorter enrollment time in real life
usage. Hence, speaker models using 60, 90, and 120 seconds of training data per
speaker were trained and tested. For each of these models, the steps are as
follows:

1. For each speaker in the cleaned corpus, determine whether enough speech data
is available for training the given speaker. If a speaker has spoken for less
than 60, 90, or 120 seconds throughout the entire corpus, they are excluded
from training *and* testing.
2. If enough data is available, set aside the first 60, 90, or 120 seconds of
speech for training and the rest for testing. Note that this may be problematic
when speakers' voices change throughout the meeting, or when the first minute
of speech has insufficient tonal variation.
3. Run maximum a priori (MAP) adaptation against a suitable universal
background model (UBM) for all speaker training data, resulting in a final
speaker model containing all speakers. As an appropriate separate Indonesian
language corpus was unavailable, the UBM was pre-built from a different source.
4. Evaluate the speaker model using the test data set aside in step 2 by
calculating the per frame speaker identification accuracy. The results of this
step are detailed in Table 2.

Training data (s)   SER (%)   # Speakers
----                ------    ----
30                  49.12     223
60                  21.71     201
90                  13.62     178
120                 13.21     157
150                 10.15     138
180                 11.65     123

There is a difference between speaker model training on a corpus and training
through speaker enrollment. When training on a corpus, as the amount of
training data per speaker is increased, a larger proportion of speakers can no
longer be trained and tested. This is noted in Table 2, which notes the number
of speakers trained and tested. Also, whereas in this case the speech taken for
each speaker's training is from the very beginning of their turns, in speaker
enrollment we have the opportunity to obtain speech according to our specific
needs. For instance, as speakers can often be discriminated from the sound
of their vowels, we may want to provide a sentence containing mostly vowels
for enrollment, which is not possible in corpus training without utilizing a
phoneme recognizer beforehand. In this way, it is possible to obtain better
results in less time through enrollment.

## 4.2 Baseline System

The baseline system utilizes the standard LIUM toolkit setup to implement
offline speaker diarization using data obtained from the corpus discussed in
Section 3. This process has been covered in depth in *ref* and *ref*, with the
relevant configuration for this experiment as follows:

1. Feature extraction with 12 Mel-Frequency Cepstral Coefficients (MFCC) in
Sphinx format from 16 kHz audio files with additional energy, delta, and
delta-delta information calculated during pre-processing. Feature warping,
cepstral mean normalization (CMS), and variance normalization are applied on a
300 frame sliding window for robustness purposes.
2. Initial speaker segmentation, followed by classification of speech, music,
and silence segments.
3. GLR-based segmentation followed by linear and hierarchical clustering of
resulting speaker segments.
4.

The system is evaluated by decoding and identifying the corpus assembled in
Section 3, with the results displayed in Table 3. It should be noted that
because the speaker model itself is built from this data, portions of the
training data are evaluated by the system. This equally applies to the online
system.

## 4.3 Proposed Method

The online system implements a simple voice activity detector (VAD) to detect
the beginning and ending of utterances, splitting the incoming audio stream
into a separate segment whenever voice activity is not detected. LIUM toolkit
is then used to run speaker identification on the segment against the speaker
model obtained in Section 4.1. For the current utterance, the system outputs
the most likely speaker calculated over a sliding window. The online system is
illustrated in *Figure 3*.

![Data flow diagram for proposed system](dfd_online.png)

In detail, the process can be broken down into the following steps:
1. Analyze the incoming audio stream to detect voice activity.
2. When a portion of the signal is voiced, keep the samples in a separate
buffer.
3. When a silent/non-voiced portion of the signal is encountered, if the
silence is preceded by a voiced portion and if the length of silence exceeds a
specified threshold, write the buffer to a file. This is our speaker segment.
4. Run speaker identification on the segment against the speaker model obtained
in Section 4.1, storing the likelihood of *all* speakers.
5. Take all speaker predictions from the previous *n* segments or *t* time,
whichever is shorter, and calculate the average score for each speaker in this
period. The highest scoring speaker is output as the prediction for this
segment.

The results of the baseline and proposed systems for the 120 second speaker
model is detailed in Table 3.

Method      SER (%)
---------   ------
Baseline
Proposed


# 6 Conclusion
