# Ray - Autoscaling Cluster for Marin Data Processing

[Ray](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) provides a simple interface
for programmatically spinning up compute and scheduling/running jobs at scale. For Marin, we structure our data 
processing jobs to be Ray compatible, so we can quickly parallelize across different nodes.

At a high-level, this directory provides all setup scripts and configuration files for standing up a Ray Cluster and
interacting with it to do lightweight monitoring and reconfiguration. The architecture of our cluster is as follows:

+ **Head Node**: A *persistent* (on-demand) 
  [`n2-standard-8` GCP VM](https://cloud.google.com/compute/docs/general-purpose-machines) with 8 CPUs and 32 GB of
  RAM, and a 200 GB disk.
+ **Worker Nodes**: An autoscaling number of TPU v4-8 VMs; a minimum of 4 VMs will be kept alive at all times, with a
  maximum of 64 VMs alive at once (we can increase this number). 

## Quickstart -- Structuring Code & Running Jobs 




## Setting Up a Ray Cluster

TODO (siddk, dlwh) -- Shouldn't require changing / updating all that often...


## Ray Cluster Utilities



## Ray Jobs & Special Cases


