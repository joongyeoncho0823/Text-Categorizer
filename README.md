# Text-Categorizer

This text categorizer uses the Rocchio / TF*IDF method to predicts which category an article belongs in, in a known,
exhaustive list of categories. 

This project was developed on a Windows system using Python3.

To run the program, simply run the following command in a terminal in the directory with the file: python project.py

In order to run this program, you must have nltk installed, and must also run some additional downloads.
The commands to run the downloads are written at the top of the project.py file, and should uncomment 
the lines of downloads. After downloading it once, you can comment it out again.

The system uses NLTK's word_tokenize for tokenizing words and punctuation. 

Some optional features that were included was case insensitivity (all letters are uncapitalized), NLTK's stop list and Porterstemmer.
Some additional features includes ignoring tokens that are numbers, and tokens that contain an apostrophe ('), since 
these tokens were often in the form of " 's " or " 're" from words such as "Steven's" or "they're". The additional features
had predicted 1~2 more articles correctly, so was included in the final implementation.

The system's performance was tested on the second and third data set by dividing the training set into a train and test set,
with a 3:1 ratio.


In order to analyze the algorithm, use the Perl analyze file using the following syntax:  perl analyze.pl <corpus1_output.txt> <corpus1_test.labels> 
