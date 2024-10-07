import multiprocessing
from playsound import playsound

p = multiprocessing.Process(target=playsound, args=("Omar.mp3",))
p.start()
input("press ENTER to stop playback")
p.terminate()


# Source: https://stackoverflow.com/questions/57158779/how-to-stop-audio-with-playsound-module
