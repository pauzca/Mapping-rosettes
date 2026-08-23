
# Mapping rosettes

This repository is part of the paper *Mapping post-fire mortality of paramo giant rosette plants using artificial intelligence on multispectral drone imagery* [].

You will find three notebooks to guide you step by step on how to do:

- `geospatial_dl_pipeline.ipynb` create a detection datasets from an orthomosaic, run a detection model and convert predictions back to geospatial format

- `bboxes_to_sam2_masks.ipynb` predict segmentation masks with SAM2 using bounding boxes as prompt, and save them in a geospatial format. 

- `plant_health_classification.ipynb` extract vegetation indices inside the segmentation masks and find a threshold for classification. 

I hope the code in each notebook can help you create and tailor a pipeline for your own needs applying deep learning models to drone data for vegetation monitoring. 

Additionally I include two small datasets, one in high resolution and one in low resolution for testing the code. As well as the weights for the YOLOv11 model trained for giant rosette plant detection. 

For many of this steps I used the package [**geodataset**](https://github.com/hugobaudchon/geodataset) 🌳.

I had initually written my own code which you can find in the archive folder of this repository, but I find geodataset to work very well and it is way cleaner solution. 
