# face-match-opencv

## Activate env
```
.\venv\Scripts\activate.bat
```

## Install dependencies
```
pip install -r requirements.txt
```

## Run
Using learned data.dat
```
python main.py -t WEBCAM 
```
Generating new learned data
```
python main.py -t WEBCAM -d c:\images_dir -o data.dat
```
