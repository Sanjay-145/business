### BUSINESS MEETING SUMMARIZATION
Business meeting summarization is a cutting-edge application of deep learning in the corporate world. It leverages artificial intelligence to automate the laborious task of condensing lengthy meetings into concise, informative summaries. 

By harnessing advanced natural language processing techniques, this technology identifies key discussion points, action items, and critical insights, ensuring that stakeholders can swiftly grasp the meeting's essence without the need to sift through extensive transcripts.

This innovation not only enhances productivity but also fosters efficient communication and decision-making within organizations. With the power of deep learning, businesses can streamline their operations and stay ahead in today's fast-paced, information-driven landscape.

### Hardware Requirements
The hardware requirements for the implementation of the proposed cosmetic product comparison system from handwritten images are outlined below:

### High-Performance Workstation:
A workstation with a multicore processor (e.g., Intel Core i7 or AMD Ryzen 7) for parallel processing.

### Graphics Processing Unit (GPU):
A dedicated GPU (e.g., NVIDIA GeForce RTX series) for accelerated computations, especially for deep learning tasks.

### Memory (RAM):
Minimum 16GB of RAM to handle the computational demands of OCR and image processing tasks.

### Storage:
Adequate storage space (preferably SSD) to accommodate large datasets and model files.

### High-Resolution Display:
A high-resolution 5 for detailed image analysis and visualization.

### Software Requirements
The software requirements for the successful deployment of the cosmetic product comparison system are as follows:

### Operating System:
A 64-bit operating system, such as Windows 10 or Ubuntu, for compatibility with modern deep learning frameworks.

### Development Environment:
Python programming language (version 3.6 or later) for coding the OCR
system.

### Deep Learning Frameworks:
Installation of deep learning frameworks, including longformer, to leverage pre-trained models and facilitate model training.

### Sklearn Libraries:
Integration of transformer libraries,  to incorporate the existing knowledge of the summarizer and analyzing large text inputs.

### PROJECT ARCHITECTURE

![Sitemap Whiteboard in Green Purple Basic Style](https://github.com/Sanjay-145/business/assets/75235426/0c211e62-9510-42b3-82af-b07a0b25f744)

### PROGRAM
### IMPORTING DATA
```
import os
import shutil
# Path to the root folder
folder_path = 'ami_meetings/'
# Paths to subfolders
abstractive_path = os.path.join(folder_path, 'abstractive')
transcripts_path = os.path.join(folder_path, 'transcripts')
test_path = 'test/'
# Create folder "/test" if it does not exist
if not os.path.exists(test_path):
os.makedirs(test_path)
# Get lists of files in both subfolders
abstractive_files = os.listdir(abstractive_path)
transcript_files = os.listdir(transcripts_path)
# Check which transcripts don't have summaries
transcripts_without_summary = []
for transcript_file in transcript_files:
summary_file = transcript_file.replace('.transcript', '.abssumm')
if summary_file not in abstractive_files:
transcripts_without_summary.append(transcript_file)
# Move transcripts without summaries to the /test folder
for transcript_file in transcripts_without_summary:
source_path = os.path.join(transcripts_path, transcript_file)
target_path = os.path.join(test_path, transcript_file)
shutil.move(source_path, target_path)
print("Moved transcripts without summaries to the /test folder.")
```
### TRAIN, TEST AND SPLIT
```
import pandas as pd
from sklearn.model_selection import train_test_split
# Assuming 'df' contains the DataFrame with columns 'transcript', 'summary', and 'meeting_id'
# Perform train-val split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# Reset the index for both DataFrames
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
```
### DATA VISUALIZATION
```
plt.figure(figsize=(12,5))
plt.subplots_adjust(wspace=0.1)
# histogram of number of words in transcripts
plt.subplot(121, xlabel='number of words', ylabel='number of transcripts', title='Words in Transcripts')
sns.histplot(data=data, x='tran_word_length', stat='count', kde=True, bins=20, hue='split', palette=custom_palette)
plt.gca().set_xticks(np.arange(0, data['tran_word_length'].max(), 2e3))
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().grid(True, alpha=.6)
# histogram of character lengths in transcripts
plt.subplot(122, xlabel='number of characters', ylabel=' ', title='Characters in Transcripts')
sns.histplot(data=data, x='tran_char_length', stat='count', kde=True, bins=20, hue='split', palette=custom_palette)
plt.gca().set_xticks(np.arange(0, data['tran_char_length'].max(), 1e4))
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().grid(True, alpha=.6)
```
### SUMMARIZATION
```
import torch
from transformers import pipeline
hf_name = 'pszemraj/led-base-book-summary'
summarizer = pipeline(
"summarization",
hf_name,
device=0 if torch.cuda.is_available() else -1,)
```
### OUTPUT

![output1](https://github.com/Sanjay-145/business/assets/75235426/7ec6761a-c2ae-4c8c-a6ce-ca5acb33fce1)
![output2](https://github.com/Sanjay-145/business/assets/75235426/bc647753-6399-4ae2-8dec-72ad5040554e)

### RESULT
The project on "Business Meetings Summarization Using Deep Learning with Longformer" proves the efficacy of advanced natural language processing. Leveraging the Longformer transformer model, the project successfully addresses the challenge of summarizing lengthy business meetings, providing concise and informative insights. This approach not only enhances efficiency but also facilitates effective decision-making. The project highlights the potential of deep learning to revolutionize information extraction, offering a valuable tool for streamlined business communication. Future research could focus on domain-specific fine-tuning, multilingual capabilities, and real-time summarization, promising continued advancements in optimizing professional interactions through intelligent systems.
