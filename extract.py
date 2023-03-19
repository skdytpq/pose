import tarfile
import os
fname = '/video'  # 압축 파일을 지정해주고   
for i in os.listdir(fname):
    if 'tgz' in i:
        ap = tarfile.open(os.path.join(fname,i))     # 열어줍니다. 

ap.extractall('extract_video')        # 그리고는 압축을 풀어줍니다. 
# () 안에는 풀고 싶은 경로를 넣어주면 되요. 비워둘 경우 현재 경로에 압축 풉니다. 
 
ap.close()                  

