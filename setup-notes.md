

# setting up google drive locally

1. install google drive 
```
$ sudo add-apt-repository ppa:alessandro-strada/ppa
$ sudo apt-get update
$ sudo apt-get install google-drive-ocamlfuse
```
2. launch google-drive-ocamlfuse (and grant permissions)
```
$ google-drive-ocamlfuse
```
If successful, will see "Access token retrieved correctly.

3. make directory for google drive
$ mkdir ~/edgetracking-googledrive


4. mount google drive to the directory
$ google-drive-ocamlfuse ~/edgetracking-googledrive


