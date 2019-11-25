from pytube import YouTube
from farhad.Farhadcolor import  tcolors,bcolors

if __name__ == "__main__":
    print(tcolors.BLUE+tcolors.BOLD)
    url = str(input('write_url: ')).encode('utf-8').decode('utf-8') 
    save_path = '/Users/apple/Downloads'.encode('utf-8').decode('utf-8') 
    print(tcolors.ENDC)
    video = YouTube(url).streams.first().download(save_path)
    print(tcolors.BLUE+tcolors.BOLD)
    print(video.title)
    print(video.thumbnail_url)
    print('*** DONE! ***')
    print(tcolors.ENDC)




