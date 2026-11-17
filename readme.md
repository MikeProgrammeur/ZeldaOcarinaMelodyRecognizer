⚠️WORK IN PROGRESS⚠️

# 🧝 Zelda Ocarina Melody Recognizer 🎤🎶​
## Intro
The new remake of Legend of Zelda Ocarina of Time on Nintendo Switch 2 added a feature allowing the player to sing in real life in order to play ocarina in game. You can hum, sing or whistle according to the trailer. Given that I have no Switch 2 nor good PC, I am building a cheap and old fashioned melody recognizer that uses the PC microphone and no deep learning to recognize one of the 12 melodies from the game. The algorithm is tailored to work with whistle and real instruments because tone estimation is quite hard for huming, especially without modern machine learning.\
The idea is that you launch Zelda OOT on your favourite emulator and simultaneously you run the main python script that listens through the microphone constantly, if the loop recognize a song, then a sequence of synthetic input is executed and therefore detected by the emulator, this way you can ✨control the game, with your voice✨.

I want to precise that AI help me, but only online chatbot for hints or for small parts of function. No agent or code assistant like codex or Claude code have been used, every single step and decision come from my mind. I coded for the pleasure of solving a problem, with the help of chatbot but I didn't watched an agent produce AI slop to fill my GitHub with another Repository.

# Tutorial
## Run the process

## Ocarina Tabs
I have not found an image with all the 12 ocarina melodies from Zelda Ocarina of Time, I have found some vibe coded website with some melodies well organized but never all 12 at the same place.
So I decided to create my own sheet, you can find it at this path `OcarinaTabs/The Legend of Zelda Ocarina of Time - Ocarina Melodies.jpg`, I used [4] and paint.net to create the visual, the ocarina tabs are found at [5].\
Here is the ocarina cheat sheet :

<p align="center">
  <img src="./OcarinaTabs/The Legend of Zelda Ocarina of Time - Ocarina Melodies.jpg" alt="Description" width="600">
  <br>
  <em>Figure : Tabs for the 12 ocarina melodies from Zelda Ocarina of Time.</em>
</p>

## Python requirements
TODO

## Input configuration
TODO

## Research and development
### Algorithm sketching
#### Note sequence estimation
In this section I am going to briefly explain how the algortihm works and tell you which .ipynb notebook you should read and in which order, especially if you want to understand deeply how I came to the final algorithm and how I developed it.\
First of all I recorded myself whistleling the Zelda Lullaby, and I made the notebook ZeldaLullabySungAndWhistledAnalysis.ipynb. (All the audio I produced is in .wav format cause ffmpeg and python do not f*****g want to work together on my laptop, maybe the problem is the guy between the keyboard and the screen 🤔).\
This notebook is the core of my research, the goal was to make an algorithm that works on this melody, but with enough tunable parameters so it can work on every melodies.\
The first step consists of cleaning the audio signal, by setting to a true zero value every silence in the recording. By doing that we ensure that no extrapolated note is recognized during a silence.\
Then the goal is to turn the temporal signal into a frequential signal and identify the prevalent frequency at each time, in reality the frequency can not be estimated at each time due to Heisenberg inequality, however we can still estimate it on short periods.\
Basically we a going to take some small chunk of the signal and perform a FFT on it, this is exaclty the behavior of Short Time Fourier Transform. Then for a temporal chunk we are simply going to take the argmax of its Fourier Transform, we will consider this frequency as the pitch for the whole chunk. If chunk are small enough saying that the frequency is constant during the chunk is not a huge approximation. We do this for every single temporal chunk. This gives a time series of pitches.\
After this rough pitch extraction we will clean it a bit, first we convet it to midi and then we apply a median filtering to remove pikes and inacurate classifications. We convert it to midi so the difference between two notes is always the same, no matter which octave they belong to.\
Now we basically have a piecewise constant signal and we want to detect the index of the break points between the different plateaus. To do so we use the python Library called Rupture [1]. This will cleverly detect frequency changes, we just have to take the average of the estimated pitch between each pair of consequent break points and this is how we go from an audio signal to a sequence of notes.

#### Melody comparison
The original goal was to know if the audio contains a known melody, fortunately it is easier to identify it in a sequence of note rather than in a raw audio signal.\
We need to have a list of the 12 songs, [2] lists those melody as sequences of N64 Controller Button, the following image from [3] shows the conversion from N64 Button to tone :

<p align="center">
  <img src="https://static.wikia.nocookie.net/zelda_gamepedia_en/images/c/c5/Ootpitches.png" alt="Description" width="600">
  <br>
  <em>Figure 1: Conversion N64 Button to tone.</em>
</p>

I stored those melody names paired with the note sequence into a dictionary. I have done that in the notebook ocarinaMelodies.ipynb, then I converted this sequences of N64 buttons into different formats : English notation, Fixed Do key notation, Hz and MIDI. All those converted song were also store into corresponding dictionaries in the notebook.\
Finally I exported all the produced dictionaries as text and paste them into the file songDictionaries.py, this way if I want to get the melodies in any format I don't need to compute it I can simply use `from songDictionaries import {The dictionary I need}`.\
Actually I created one more dictionary the differentiate of MIDI melodies.\
Instead of matching the exact sequence of MIDI notes, we compare the sequence of pitch intervals (MIDI note differences), making the recognition invariant to the absolute pitch of the performance.\
To detect a melody in a sequence of notes we just do a sliding window on the sequence and compute the norm of (subsequence - melody), I tried different losses/norm such as mae, rmse, max and huber.\
Then among all widows we take the one which has the lowest loss and we compare this loss to a threshold to decide wether there is a melody played or not.

### Algorithm tuning and integration tests

### Development of main.py

# Credits
## Bibliography
[1] Rupture python library : https://github.com/deepcharles/ruptures, btw I had the opportunity to attend Charles Truong's tutorials during my studies, I highly recommend that you check out his work.

[2] Ocarina melodies in-game button sequences : https://www.zeldadungeon.net/wiki/Ocarina_of_Time_Songs.

[3] Conversion from in game button to tone : https://zelda.fandom.com/wiki/Ocarina_of_Time_(Item).

[4] Ocarina-tabs editor : https://panzi.github.io/ocarina_tabs/.

[5] Game's melodies Ocarina-tabs : https://traxitabs.tumblr.com/search/requiem.

[] Ocarina C5-tone sample : https://freewavesamples.com/kawai-k5000w-ocarina-c5.







## My personnal contributions :
*.md

*.py

*.ipynb

/utils/*.py

Data/ZeldaLullabyForPoc/* 

Data/AllSongsWhistled/*

OcarinaTabs/* (The work is mine but I used a tool from github to produce it, if you want more informations, look at OcarinaTabs/Infos.txt)