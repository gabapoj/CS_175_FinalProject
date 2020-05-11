List of all files you submitted:

Asst6.cpp

Note the platform you used for development (Windows, OS X, ...):

Windows

Provide instructions on how to compile and run your code, especially if you used a nonstandard Makefile, or you are one of those hackers who insists on doing things differently.

Everything is normal!

Indicate if you met all problem set requirements (more importantly, let us know where your bugs are and what you did to try to eliminate the bugs; we want to give you as much partial credit as we can).

We believe we met all requirements.

Provide some overview of the code design. Don't go into details; just give us the big picture.

Task 1 was implemented with a list<vector<RigTForm>> for the script, an iterator for the current frame, and a vector<shared_ptr<SgRbtNode>> for the map. 
Almost all thw implementation was getting iterators to behave and be able to copy RigTForms from the map to the current frame and vice versa. 
Our file format is .txt files with a header line explaining the size of the file and view speed, and then as many lines of 7 floats (Describing rbts) as RBTs are required to fulfil the header.

Task 2 was implemented practically from the book. 
Task 3 was practically straight off of the pdf. 

Let us know how to run the program; what are the hot keys, mouse button usage, and so on? Describe steps or sequences of steps the TF should take to test and evaluate your code (especially if your implmenentation strays from the assignment specification).

To load a file, name it 'input.txt'. Output files from the program are named 'output.txt'. Otherwise, keys are as the assignment prescribed. 

Everything was as prescribed except the extra stuff

Finally, did you implement anything above and beyond the problem set? If so, document it in order for the TFs to test it and evaluate it.

To increase the frame speed change functions (+/-), hold space. Then frame speed will change 10x per press instead of 1x.
To loop the animation, press 'l'.
Also, our 'input.txt' is a fun animation. Please play it (press i, then l or y). 