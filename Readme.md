# Multimodal Humor Detection
This Project aims to detect humor within the dataset. We took a multimodal approach to the problem and detect cues for humor within the video clip through visual, audio and text modalities.

## Dataset:
We used the UR-FUNNY and UR-FUNNY2 Datasets (insert link here). The datasets contain a set of video clips collected from TED talks, labeled by human from the lab to see if there is humor within each clip. Clips vary in length and quality. Some videos contain audios that are inaudible and just white noises, some contain visual feature that are just from the powerpoints instead of the speaker or the audience. However, we over come that by data cleaning. In the repo we do not include the dataset due to its size, but it can be downloaded by running

<pre>```bash
python Feature_Extraction_Scripts/download_dataset.py
```</pre>

## Data Cleaning:
For our data cleaning, we filtered the clips through cross comparison between three modalities. For our visual features, we filter them by recognizing if the face exits in the clips or not. If it exists, we claim that the video contain enough visual feature to say if there is a cue for humor or not. For audio, we cross compare with text extracted from the clip to see if the audio contain actual words or gibberish. To do data cleaning, follow the steps below:

1. Run 535_data_filtering.ipynb to filter out the bad video clips

2. convert to .wav files for audio extraction and cleaning:
<pre>```bash
python Feature_Extraction_Scripts/convert_to_wav.py
```</pre>

3. Run the three scripts for audio, visual and text extraction:
<pre>
```bash
python Feature_Extraction_Scripts/extract_feature_only.py    python Feature_Extraction_Scripts/feature_extract.py    python Feature_Extraction_Scripts/text_feature_extract_ModernBERT.py
```
</pre>

Due to version controls, we have some code written in Jupyter Notebook format for easier debugging and tracking.

## Uni and Multi Modal Testing:
After the features have been extracted, just run the other notebooks for training and using the classification models. In the folder "UniModal_Testing_Scripts" are some other unimodal testings serving as a comparison for the other approaches we did. 
