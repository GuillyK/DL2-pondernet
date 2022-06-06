# PonderNet does wonders: Using PonderNet to determine the right amount of computational iterations
PonderNet is an architecture that attempts to optimize computational efficiency and accuracy. It does this by altering the loss function of any supervised architecture.

## Description
PonderNet is an architecture that allows an architecture to ponder automatically, that is, decide when to 'halt' the recurrent function from being applied to the current prediction again.  For example, when considering the MNIST digit dataset, PonderNet might decide to use fewer steps when classifying the digit six, as it has distinctive features, but might use more steps when classifying the digits one and seven, as they have fewer distinctive features.

### Executing program

* Download the files from GitHub
* Run the PonderNet network by calling 
```
python PonderNet.py --arguments

--dataset: str, select dataset. default is MNIST, select from "MNIST", "FMNIST" or "CIFAR10"
--rotation: bool flag, use rotations.
--no-rotation: bool flag, don't use rotations.
--ponder: bool flag, use ponderloss.
--no-ponder: bool flag, don't use ponderloss.
```


## Authors

Alex Labro\
11872470\
alex.labro@student.uva.nl

Guilly Kolkman\
11822465\
guilly.kolkman@student.uva.nl

Joris Hijstek\
11876980\
joris.hijstek@student.uva.nl