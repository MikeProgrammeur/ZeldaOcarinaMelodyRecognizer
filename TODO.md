# TODO
[x] Creating the JSON for key configuration
[x] Create a confirmation routine that wait x seconds for the user to press a key to confirm a song
[x] Create a function that can simulate any song according to the key configuration
[x] Add the confirmation routine and the melody input simulation function in the main loop
[x] Add a melody-adaptative threshold because Zelda Lullaby suffers from problematic false positives
[x] Create a script that detects ocarina buttons of a controller, to resimulate them as keyboard input to send them to the emulator, because simulate keyboard is easier than simulate game-controller.
[x] Correct debug mode in main.py
[x] Redefine the silence detection, make it robust against full noise signal, using spectral flatness detection
[x] Write key configuration tuto for compatibility with emulator

# Priority
[] Change the main script from permanent listening, to press-to-listen (click space to begin recording, click space to stop, click enter to accept the suggested song for example)
[] Write the xboxToProject64SynthInput.py file
[] Finish the section R&D details in the readme

# Bonus
[] Create a fancy app for key configuration
[] Find a way to make the script listen only when ocarina menu is opened in game