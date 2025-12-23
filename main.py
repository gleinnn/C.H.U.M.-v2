import ollama
from playsound import playsound as ps

import torch
from TTS.api import TTS


client = ollama.Client()
device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

#ollama parameters
system = "You are a helpful assistant named C.H.U.M. Be casual, not too formal. Create very short, but concise responses."
model = "smollm2"
inp = ""

while inp != "exit":
  inp = input("You: ")

  response = client.generate(model=model, prompt=inp, system=system)
  print("C.H.U.M.:" + response.response)

  tts.tts_to_file(
    text=response.response,
    language="en",
    speaker="Craig Gutsy",
    file_path="response.mp3",
    speed=1.3
  )

  ps("response.mp3")