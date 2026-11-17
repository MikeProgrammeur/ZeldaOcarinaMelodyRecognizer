from pyautogui import press, typewrite #Source - https://stackoverflow.com/a/43567521, Posted by CornSmith

from typing import Dict

import keyboard
import threading
import time

import json

import numpy as np

from utils.songDictionaries import song_dict_button

def init_button_to_key_dict(jsonPath : str = "keyConfiguration.json"):
    try:
        with open(jsonPath, "r") as f:
            button_to_key_dic = json.load(f)
            for e in ['A', 'Up', 'Right', 'Down', 'Left']:
                if button_to_key_dic.get(e) is None:
                    return None
            button_to_key_dic = {k:button_to_key_dic[k] for k in ['A', 'Up', 'Right', 'Down', 'Left'] if k in button_to_key_dic}
            return button_to_key_dic
    except:
        return None
    
def init_song_to_key_dict(song_to_button_dict, button_to_key_dict):
    dic_res = {}
    for _, (key, value) in enumerate(song_to_button_dict.items()):
        str_temp = ""
        for i,e in enumerate(value):
            str_temp += button_to_key_dict[e]
        dic_res[key] = str_temp
    return dic_res

def read_confirmsong_key(jsonPath : str = "keyConfiguration.json"):
    try:
        with open(jsonPath, "r") as f:
            button_to_key_dic = json.load(f)
            return button_to_key_dic.get('ConfirmSong')
    except:
        return None

def waitForKey(key: str, timeout: int) -> bool:
    event = threading.Event()

    keyboard.on_press_key(key, lambda _: event.set())

    try:
        return event.wait(timeout / 1000)
    finally:
        keyboard.unhook_all()
        
def simulateSong(song_name : str, song_to_key_dict : Dict, print_log : bool = False) -> None:
    if print_log:
        print(f"playing the song {song_name}")
    sequence = song_to_key_dict.get(song_name)
    if sequence is not None:
        typewrite(sequence)
    else:
        print(f"Song '{song_name}' not recognized.")

if __name__ == '__main__':
    print('Reading the JSON')
    confirm_song_key = read_confirmsong_key()
    if confirm_song_key is None:
        raise ValueError('Confirm song key is not correct in the json')
    print(f"Waiting for confirmation : you have 5 seconds to press {confirm_song_key}")
    res = waitForKey(confirm_song_key, 5000) 
    if res :
        button_to_key_dict = init_button_to_key_dict()
        if button_to_key_dict is not None:
            song_to_key_dict = init_song_to_key_dict(song_dict_button, button_to_key_dict)
            print('Simulating a song')
            simulateSong('Zelda Lullaby', song_to_key_dict)
            print('Song has been simulated successfully')
        else:
            raise ValueError('One of the N64 Button key is not correct in the json')
    else:
        print("Time limit reached, song playing will be cancelled")
        

