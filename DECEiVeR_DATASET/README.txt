%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%						README -- DECEiVeR dataset
%
% This short guide to a snippet of the DECEiVeR dataset, covers the following 
% topics:
% (1) Note.
% (2) General information about the dataset.
% (3) Structure of the dataset.
% (4) Usage.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------------------------------------------------------------------------
(1) Note:
-------------------------------------------------------------------------------
The DECEiVeR dataset contains data from first 11 subjects


-------------------------------------------------------------------------------
(2) General Information:
-------------------------------------------------------------------------------
The Dataset Acting Emotions Valence and Arousal (DECEiVeR) dataset was developed through a study involving 11 professional actors to collect physiological data for emotion-driven acting. This ultimately aims to investigate the correlation between biosignals and emotional expression for Affective Computing.

DECEiVeR is a comprehensive collection of various physiological recordings meticulously curated to facilitate the recognition of a specific set of five emotions.

-------------------------------------------------------------------------------
(3) Structure of the Dataset:
-------------------------------------------------------------------------------

The root folder of the DECEiVeR dataset, where this README file is located, has the
following subfolders:

(a) /DECEiVeR_raw
(b) /DECEiVeR_resampled
(c) /DECEiVeR_session
(d) /DECEiVeR_arouval
(e) DECEiVeR_features
 
Each of these subfolders and any further subfolders within them also contain
README files for more information. As such, only a short description of
these subfolders is provided here:

(a) /DECEiVeR_raw - contains log data as CSV files consisting of nine variables, each     occupying its own column.

(b) /DECEiVeR_resampled - this folder contains other information about the experiments. For example, information about the participants, the sequence in which they
    watched the videos, etc.

(c) /scripts - contains scripts that allow the user to undertake/verify the 
    steps required for converting raw data to the processed data contained in
    the interpolated and non-interpolated folders. 

-------------------------------------------------------------------------------
(4) Usage:
-------------------------------------------------------------------------------
For researchers seeking to conduct downstream analysis with the provided data, we recommend utilizing the interpolated data due to its consistent size across different participants and individual videos. 

However, we also provide the non-interpolated data for researchers who prefer to use a different interpolation approach or have specific reasons for using the original data. While researchers are welcome to work with the raw data, it is important to note that the presented approach for augmenting video-IDs to the physiological and annotation data should not be altered, as doing so may lead to inconsistent results in subsequent analyses.
   