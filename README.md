# Diffusion-Based-AD

In this project different approaches of anomaly detection with Diffusion Models are explored and implemented. <br>
With a focus on industrial anomaly detection the test are currently run for the [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) but can easily setup for your own data.

### Setup
Depending on your preferences you can either run the [environemnt.yml](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/environment.yml) to create a conda environemnt, or the [requirements.txt](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/requirements.txt) in your virtual env. <br>
The project is configured to use Python 3.10, and uses the huggingface diffusers library as a backbone.

### Run experiments
You can either run individual parts you are interested in via their respective scripts (i.e. training the diffusion model, training the feature extractor, evaluate the model etc.) or use the train-tune-eval script, which can be also be adjusted via cli parameters to your needs.

For the most simple approach you can run the following command: <br>
`python train_tune_eval.py --diffusion_checkpoint_dir
checkpoints/hazelnut
--run_id
hazelnut
--diffusion_checkpoint_name
epoch_1000.pt
--extractor_path
checkpoints/hazelnut_01.pt
--item
hazelnut
--dataset_path
PATH/TO/YOUR/DATASET
--plt_imgs
--recon_weight
.1
--eta
.1
--pxl_threshold
0.029
--feature_threshold
0.33`

You might want to add `--skip_threshold` for better results, as the static threshold estimation for the difference map is not as good atm.
This will train the diffusion model, train the extractor for the feature extractor which is used by the diffmap and evaluate it on the test-set.
Depending on the dataset you use you'll probably want to tune the parameters used during [inference](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/inference_ddim.py#L52) to get better results.

You can also individually run the [train-script](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/train_ddim.py), [threshold-evaluation script](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/evaluation.py), [feature-extractor training script](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/train_extractor.py) or the [evaluation script](https://github.com/NilsB98/Diffusion-Based-AD/blob/master/inference_ddim.py).


### Add custom datasets
To use custom data you can either implement your own data loader or structure your data like the MVTec Dataset.<br>
The folder-structure would the look like this:<br>
├───bottle <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───ground_truth <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───broken_large <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───broken_small <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───contamination <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───test <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───broken_large <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───broken_small <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├───contamination <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───good <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───train <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───good <br>


### Related Papers and Features
* [Anomaly Detection with Conditioned Denoising Diffusion Models](https://arxiv.org/pdf/2305.15956v1.pdf): This paper displayed the idea to condition the diffusion model on the original image, i.e. interpolate the latent vector with the initial image, as well as using a feature-based difference map. <br>

* Image Patching: To keep the initial resolution (possibly for really small anomalies) the image can be separated into a grid of patches, the detection evaluated on each path and afterwards stitched back together. 

* Combining of different anomaly maps (i.e. pixel-level and feature level anomaly maps)

### Roadmap
1. Evaluate the full MVTec Dataset
2. Implement [LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection](https://arxiv.org/abs/2307.08059)