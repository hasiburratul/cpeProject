# Importing relevant libraries and modules
import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

# This is a script to import all images from a folder at once using the os library.
# We are importing all the images in one list and image names in another list
path = 'Images'  # Just defining a normal string variable
images = []  # A list containing all the images
imageNames = []  # A list containing all the image names
imagelist = os.listdir(path)  # Collecting all the image names in the specified directory, aka images


# This is a script to update the images names excluding the image type from the names.
for ids in imagelist:  # Run all the elements of the list
    currentImg = cv2.imread(f'{path}/{ids}')  # identifying the file name as they are stored as elements in the list
    images.append(currentImg)
    imageNames.append(os.path.splitext(ids)[0])  # split-'ext' i.e. splits the file name from the extension in each names in path




# This a function to find encodings of the images from the list of images
# and create a corresponding encoded list for the images.
def calEncodings(images):
    encodedList = []  # A list to store the image encodings
    for img in images:  # Running through all the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting the image into RGB
        encode = face_recognition.face_encodings(img)[0]  # Finding the encoding using the fac_recognition library and store to encode
        encodedList.append(encode)  # Adding the encoding to the encoding list
    return encodedList  # Returning the encoding list

# This is a function to collect the time stamp
def timeStamp(id):
    with open('timeStamp.csv','r+') as file: # Opening the file in read and write mode
        entryList = file.readlines()  # Reading the line of the file
        idList = [] # A list to store all the entries
        for line in entryList: # A loop to collect the entry ID information from the file
            entry = line.split(',')
            idList.append(entry[0])
        if id not in line: # If the entry is not registered in the file
            currenttime = datetime.now() # Store the current time
            time = currenttime.strftime("%H:%M:%S") # Store the time in organized format
            file.writelines(f'\n{id},{time},') # Write the entry ID and time in the file
        if id in line: # If the entry is registered in the file
            if id != idList[-1]: # Check the last stored entry
                currenttime = datetime.now() # Store the current time
                time = currenttime.strftime("%H:%M:%S")  # Store the time in organized format
                file.writelines(f'\n{id},{time}') # Write the entry ID and time in the file


# To initiate the encoding process
KnownIds = calEncodings(images)
print('Encoding Complete')

# A loop to run the webcam to create a video object.
cap = cv2.VideoCapture(0)

print("Starting....")

# A while loop to capture the image frames from the webcam and create encodings
while (1):
    success, img = cap.read()  # Just capturing frame from the webcam
    frameSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resizing the image to 1/4th of its original
    frameSmall = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting the image to RGB

    # Detecting the faces in the image
    currentFrame = face_recognition.face_locations(frameSmall)
    # Finding encoding of the detected faces from the image/identifying who they are
    encodeCurrentFrame = face_recognition.face_encodings(frameSmall, currentFrame)

    # Matching encodings of our known faces to webcam generated images to find the match
    for encodeFrame, faceLocation in zip(encodeCurrentFrame, currentFrame):
        # Comparing the images encodings
        matches = face_recognition.compare_faces(KnownIds, encodeFrame)
        # Computing the distance between two images
        idMatch = face_recognition.face_distance(KnownIds, encodeFrame)
        # Determining the matching image name based on the distances
        # Finds index of minimum element
        matchedId = np.argmin(idMatch)


        # Displaying information if images matches
        if matches[matchedId]:
            id = imageNames[matchedId].upper()
            y1, x2, y2, x1 = faceLocation
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, id, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1)

            # Calling a function to collect the time stamp
            timeStamp(id)

            # Open the file containing Positive Covid IDs
            file = open("covidPositive.txt")
            if(id in file.read()): # If the entry is in the file
                cv2.putText(img, 'Not Allowed', (x1 - 6, y2 + 25 ), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)# Print not allowed

            else: # Else
                cv2.putText(img, 'Allowed', (x1 - 6, y2 + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1) # Print Allowed

        # Displaying information if images doesn't match
        else:
            y1, x2, y2, x1 = faceLocation
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, 'Unidentified', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1)
            cv2.putText(img, 'Not Allowed', (x1 - 6, y2 + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)


        # Displaying the webcam current input
        cv2.imshow('Webcam', img)
        cv2.waitKey(1)




