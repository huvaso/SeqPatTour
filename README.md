# SeqPatTour

## Sequential Patterns-based Metrics for Better Understanding the Tourism Behaviour

**SeqPatTour** is a methodology that innovatively mines and analyses tourist behaviour through sequential pattern analysis with data from Trip Advisor.

The **SeqPatTour** methodology is meticulously implemented in Python 3.9.18, comprising of three comprehensive notebooks.

1. *tpm_python.ipynb*: It contains sentences to build the graph and extract paths from it using several methods, such as A*, BFS, and Dijkstra. It also allows extracting sequential patterns with gap constraints using the library [Seq2Pat](https://github.com/fidelity/seq2pat). Then, several metrics were extracted in this same notebook. To work correctly, it uses eight Python files stored in the directory called *sources*.
2. *osm.ipynb*: This notebook contains the functions and sentences for analyzing patterns spatially through Open Street Maps and permits the visualization of maps.
3. *tmp_figures.ipynb*: It contains functions and sentences to visualize histograms and other figures for comparing techniques.

Finally, the directory ***data*** contains the dataset used for experimental purposes. The dataset comprises two files, one storing the vertices and the other storing the edges. The directory ***sources*** contains our own libraries for reading files, pattern extraction, visualization, etc. Finally, ***mel_communes*** stores the shape files of Lille City.

## Requirements

**SeqPatTour** needs some libraries to work adequately:

* Geopandas 0.14.1
* Geopy 2.4.0
* Igraph 0.11.2
* Matplotlib 3.8.2
* Networkx 2.8.8
* Numpy 1.26.2
* Osmnx 1.8.1
* Pandas 2.1.3
* Powerlaw 1.5
* Prefixspan 0.5.2
* Scipy 1.11.4
* Seq2Pat 1.4.0
* Sequence-mining 0.0.3
* Sequential 1.0.0

## Support

Please submit bug reports, questions and feature requests [issues](https://github.com/huvaso/SeqPatTour/issues)

## Citation

TBA

## Authors 
* Hugo Alatrista-Salas
* Gaël Chareyron
* Sonia Djebali
* Imen Ouled-Dlala
* Nicolas Travers
