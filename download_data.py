import gdown

url = 'https://drive.google.com/uc?id=1RquJ8e_9d3ymBDuke4OP_IyoOAgi3-i-'
output = 'data.zip'
gdown.download(url, output, quiet=False)
