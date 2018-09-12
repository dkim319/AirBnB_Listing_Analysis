
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There code only requires the standard installation of Anaconda Python.  It will requires Pip to install the xgboost and geopy libraries.

## Project Motivation<a name="motivation"></a>

The purpose of this project is leverage the AirBnB data for the Los Angeles, CA area to answer these questions:

1. What aspects of an AirBnB drive the listing price in the Los Angeles?
2. Does the host's rating or the distance of the listing from key destinations influence price?
3. Can we use the various attributes of an AirBnB listing to predict price?

## File Descriptions <a name="files"></a>

There are 3 Python files.  There is 1 for the data preparation and analysis for the first two questions.  There are two Python files for the last question. Both contain machine learning model to predict price.

There are 3 data files.  One is the source data, another is to transfer the data between Python files and another is an interim dataset to expedite run time of the first file.

## Results<a name="results"></a>

The findings are documented in this Medium post. [here](https://medium.com/@dkim319/can-i-tell-if-an-airbnb-listing-is-overpriced-9a2d1361e3fe).


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The AirBnB was made publicly available by AirBnb. Link: [here](http://insideairbnb.com/get-the-data.html)

The code used in this project was learned from the Data Scientist Nanodegree Term 2 class.