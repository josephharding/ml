
## Deploy to CUDA enabled server
rsync -avz --partial --progress --exclude=env . toybox:~/customimage
