# legalOccupier

## Intro to algorithm
This algorithm is used to determine whether a car/truck/person is a legal occupier. Please download both legalOccupier.py and worker_classifier.py. Also please install OpenCV library before running the code.

## Inputs
To find legal Occupier, please call the function findLegalOccupier(type,ROI,trajectory,link) in legalOccupier.py. The inputs are:

type: type of trespassing. Please use the 'type' column in the dataset as the input.

ROI: Coordinate for region of interests. The input should be a list of tuple. Here is the ROI for Thomasville North/South, Ramsey, and Ashland:

Ramsey:[(10,396),(86,574),(581,283),(453,268)]

Ashland:[(424,702),(555,872),(1588,484),(1513,468)]

Thomasville South: [(34,855),(1289,643),(1534,644),(398,1076)]

Thomasvile North: [(301,620),(1391,1055),(1734,761),(535,587)].

trajectory: trajectory of the trespasser. The input of be a list of list. Please use the 'trajectory' column in the dataset as the input.

link: video link or the directory of the video. Please use the 'clip' column in the dataset as the input.

## Output
The output value is either True or False.True if a trespasser is a legal occupier and false otherwise.

