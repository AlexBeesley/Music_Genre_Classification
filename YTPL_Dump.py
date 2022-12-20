import os
import re

from pytube import Playlist

Playlists = [["Hip-Hop", "https://www.youtube.com/watch?v=JpSuinPCxBU&list=PLuDoiEqVUgejiZy0AOEEOLY2YFFXncwEA"],
             ["Jazz", "https://www.youtube.com/watch?v=vmDDOFXSgAs&list=PLnraNOoC4vjJpNRaBDPUNPaYgMRCun5tH"],
             ["Pop", "https://www.youtube.com/watch?v=C7dPqrmDWxs&list=PLGYPpIsdZKnLRU3hBKDmUBRdzVdM0rS0z"],
             ["Classical", "https://www.youtube.com/watch?v=XfEE-GoJS2E&list=PLU1vzYQWSdMvoF2HqfWe2gr0_kAbVpo7J"],
             ["Country", "https://www.youtube.com/watch?v=4zAThXFOy2c&list=PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S"], ]


def DownloadPlaylist(url, folderName):
    pl = Playlist(url)
    pl._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")
    items = len(pl.video_urls)
    if items == 0:
        raise Exception("Invalid YouTube URL.")
    if not DataPresence(folderName, items):
        print("There were " + str(items) + " tracks found in this playlist.")
        print("Initialising download...")
        for video in pl.videos:
            if video.age_restricted:
                print("Skipping age restricted download...")
                continue
            title = str(video.streams[0].title)
            if os.path.exists(os.path.join(folderName, title + ".mp3")):
                print("File already exists, skipping...")
                continue
            if title == "Video Not Available":
                print("Skipping unavailable download...")
                continue
            print("downloading: " + title)
            audioStream = video.streams.get_audio_only()
            audioStream.download(output_path=f'./{folderName}')


def DataPresence(folderName, expectedItems):
    if os.path.isdir(f"./{folderName}"):
        if len(os.listdir(f"./{folderName}")) == expectedItems:
            return os.path.isdir(f'./{folderName}')
    return False


for i in range(len(Playlists)):
    print(f"Starting downloads for {Playlists[i][0]} from {Playlists[i][1]}")
    DownloadPlaylist(Playlists[i][1], Playlists[i][0])
