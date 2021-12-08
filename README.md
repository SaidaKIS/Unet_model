# Unet_model
Unet model for solar granular segmentation

Solar granulation is the visible signature of convective cells at the solar surface. The granulation cellular pattern observed in the continuum 
intensity images is characterised by diverse structures e.g. bright individual granules of hot rising gas or dark intergranular lanes. Recently, 
the access to new instrumentation capabilities has given us the possibility to obtain high-resolution images, which have revealed the overwhelming 
complexity of the granulation, e.g. exploding granules and granular lanes. 

In that sense, any research focused on understanding solar small-scale phenomena on the solar surface is sustained on the effective identification and 
localization of the different resolved structures. 

This respository contains the initial results of our classification model of solar granulation structures based on semantic segmentation. 
We inspect the approach presented in the U-Net architecture, which uses convolutional networks for biomedical image segmentation to adapt a 
suitable fully convolutional network and training strategy for our science case. As our training set, we use continuum intensity maps of IMaX 
instrument inside \textit{Sunrise} balloon-borne solar observatory and their corresponding segmented maps, initially labelled using the multiple-level 
technique (MLT) and also labelled by hand. We performed several tests of the performance and precision of this approach in order to evaluate the versatility 
of the U-Net architecture. We found an interesting potential of the U-Net architecture to identify cellular patterns in solar granulation images reaching 
matching in overall pixels greater than 80\% on initial testings.
